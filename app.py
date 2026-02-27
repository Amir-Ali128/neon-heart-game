import base64
import gc
import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurovision-ai")


INPUT_SIZE = 320
DEVICE = "cpu"


class BrainActivityEntry(BaseModel):
    region: str
    mean_activation: float
    max_activation: float


class DetectionEntry(BaseModel):
    class_name: str
    confidence: float
    bbox_xyxy: List[float]


class AnalysisResponse(BaseModel):
    detections: List[DetectionEntry]
    brain_activity: List[BrainActivityEntry]
    decision_path: List[str]
    gradcam: str


@dataclass
class HookHandle:
    region: str
    handle: torch.utils.hooks.RemovableHandle


class NeuroVisionEngine:
    """Sequential CPU inference pipeline with activation logging and GradCAM."""

    def __init__(self) -> None:
        logger.info("Loading YOLOv8n + CLIP models on CPU")
        self.yolo = YOLO("yolov8n.pt")
        self.yolo_model = self.yolo.model.to(DEVICE)
        self.yolo_model.eval()

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.class_names = self.yolo.model.names

        self.yolo_modules = list(self.yolo_model.model)
        if len(self.yolo_modules) < 4:
            raise RuntimeError("Unexpected YOLO model layout; cannot map anatomical regions.")

        mid_idx = len(self.yolo_modules) // 3
        deep_idx = (2 * len(self.yolo_modules)) // 3

        self.region_to_layer_idx = {
            "V1": 0,
            "V2": mid_idx,
            "V4": deep_idx,
            "Inferotemporal_Cortex": len(self.yolo_modules) - 1,
        }

    @staticmethod
    def _to_bchw_tensor(image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(DEVICE)

    @staticmethod
    def _to_cv2_rgb(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR), dtype=np.uint8)

    @staticmethod
    def _activation_stats(tensor: torch.Tensor) -> Tuple[float, float]:
        detached = tensor.detach().float()
        return float(detached.mean().item()), float(detached.max().item())

    def _register_forward_hooks(
        self, layer_map: Dict[str, int], activation_store: Dict[str, torch.Tensor]
    ) -> List[HookHandle]:
        hooks: List[HookHandle] = []
        for region, idx in layer_map.items():
            module = self.yolo_modules[idx]

            def _capture(_, __, output, region_name=region):
                if isinstance(output, (tuple, list)):
                    out = output[0]
                else:
                    out = output
                if isinstance(out, torch.Tensor):
                    activation_store[region_name] = out.detach().cpu()

            handle = module.register_forward_hook(_capture)
            hooks.append(HookHandle(region=region, handle=handle))
        return hooks

    @staticmethod
    def _remove_hooks(hooks: List[HookHandle]) -> None:
        for h in hooks:
            h.handle.remove()

    def _run_yolo_detection(
        self, image: Image.Image, activation_store: Dict[str, torch.Tensor]
    ) -> Tuple[List[DetectionEntry], Optional[object]]:
        hooks = self._register_forward_hooks(self.region_to_layer_idx, activation_store)
        try:
            with torch.no_grad():
                results = self.yolo.predict(
                    source=image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR),
                    imgsz=INPUT_SIZE,
                    device=DEVICE,
                    verbose=False,
                )
        finally:
            self._remove_hooks(hooks)

        detections: List[DetectionEntry] = []
        if not results:
            return detections, None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections, result

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                DetectionEntry(
                    class_name=str(self.class_names.get(cls_id, cls_id)),
                    confidence=round(conf, 4),
                    bbox_xyxy=[round(v, 2) for v in xyxy],
                )
            )

        return detections, result

    def _run_clip_embedding(
        self, image: Image.Image, activation_store: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        inputs = self.clip_processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        activation_store["Prefrontal_Cortex"] = image_features.detach().cpu()

        for key in list(inputs.keys()):
            del inputs[key]
        gc.collect()
        return image_features

    def _compute_gradcam(self, image: Image.Image) -> str:
        grad_layer = self.yolo_modules[self.region_to_layer_idx["V4"]]
        grad_store: Dict[str, torch.Tensor] = {}

        def _fwd_hook(_, __, output):
            grad_store["acts"] = output[0] if isinstance(output, (tuple, list)) else output

        handle = grad_layer.register_forward_hook(_fwd_hook)

        try:
            inp = self._to_bchw_tensor(image)
            inp.requires_grad_(True)

            outputs = self.yolo_model(inp)
            if isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs

            if not isinstance(pred, torch.Tensor):
                raise RuntimeError("Unexpected YOLO forward output for GradCAM")

            score = pred[..., 4].max()
            self.yolo_model.zero_grad(set_to_none=True)
            score.backward()

            feats = grad_store.get("acts")
            if feats is None or feats.grad is None:
                grads = torch.autograd.grad(score, feats, retain_graph=False, allow_unused=True)[0]
            else:
                grads = feats.grad

            if feats is None or grads is None:
                raise RuntimeError("Could not compute GradCAM tensors")

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * feats).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            rgb_np = self._to_cv2_rgb(image)
            overlay = cv2.addWeighted(rgb_np[:, :, ::-1], 0.5, heatmap, 0.5, 0)
            ok, png = cv2.imencode(".png", overlay)
            if not ok:
                raise RuntimeError("Failed to encode GradCAM image")

            encoded = base64.b64encode(png.tobytes()).decode("utf-8")
            return encoded
        finally:
            handle.remove()
            gc.collect()

    def _brain_activity_from_store(self, activation_store: Dict[str, torch.Tensor]) -> List[BrainActivityEntry]:
        ordered_regions = ["V1", "V2", "V4", "Inferotemporal_Cortex", "Prefrontal_Cortex"]
        entries: List[BrainActivityEntry] = []

        for region in ordered_regions:
            tensor = activation_store.get(region)
            if tensor is None:
                mean_act, max_act = 0.0, 0.0
            else:
                mean_act, max_act = self._activation_stats(tensor)
            entries.append(
                BrainActivityEntry(
                    region=region,
                    mean_activation=round(mean_act, 6),
                    max_activation=round(max_act, 6),
                )
            )
        return entries

    def analyze_image(self, image_bytes: bytes) -> AnalysisResponse:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        activation_store: Dict[str, torch.Tensor] = {}

        detections, _ = self._run_yolo_detection(image, activation_store)
        _ = self._run_clip_embedding(image, activation_store)
        gradcam_b64 = self._compute_gradcam(image)
        brain_activity = self._brain_activity_from_store(activation_store)

        decision_path = [
            "Visual features extracted in V1",
            "Shape features processed in V2",
            "Object representation formed in IT Cortex",
            "Semantic meaning refined in Prefrontal Cortex",
        ]

        # explicit cleanup for small-memory environments
        activation_store.clear()
        gc.collect()

        return AnalysisResponse(
            detections=detections,
            brain_activity=brain_activity,
            decision_path=decision_path,
            gradcam=gradcam_b64,
        )


app = FastAPI(title="neurovision-ai", version="1.0.0")
engine = NeuroVisionEngine()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "neurovision-ai"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return engine.analyze_image(image_bytes)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
