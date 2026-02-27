from __future__ import annotations

from neurovision.interfaces import (
    ActivationExtractor,
    ExplainabilityEngine,
    Network3DVisualizer,
    ObjectDetector,
    SemanticAnalyzer,
)
from neurovision.models import (
    ActivationTensorInfo,
    BoundingBox,
    Detection,
    HeatmapArtifact,
    NetworkState3D,
    SemanticInsight,
)


class BaselineObjectDetector(ObjectDetector):
    """Production placeholder for a detector (e.g., YOLOv8, DETR, RT-DETR)."""

    def detect(self, image_bytes: bytes) -> list[Detection]:
        _ = image_bytes
        return [
            Detection(
                label="person",
                confidence=0.97,
                bbox=BoundingBox(x_min=0.14, y_min=0.12, x_max=0.67, y_max=0.95),
            )
        ]


class BaselineSemanticAnalyzer(SemanticAnalyzer):
    """Production placeholder for VLM- or LLM-assisted semantic interpretation."""

    def analyze(self, image_bytes: bytes, detections: list[Detection]) -> SemanticInsight:
        _ = image_bytes
        tags = sorted({detection.label for detection in detections})
        return SemanticInsight(
            scene_description="A person is centered in the frame with high confidence detection.",
            tags=tags,
            reasoning="Semantic context derived from object classes and spatial priors.",
        )


class GradCamExplainabilityEngine(ExplainabilityEngine):
    """Production placeholder for Grad-CAM, Score-CAM, or Eigen-CAM style outputs."""

    def generate(self, image_bytes: bytes, detections: list[Detection]) -> HeatmapArtifact:
        _ = image_bytes, detections
        return HeatmapArtifact(
            algorithm="grad_cam",
            uri="s3://neurovision-artifacts/heatmaps/request_id/base_heatmap.npy",
            overlay_uri="s3://neurovision-artifacts/heatmaps/request_id/overlay.png",
        )


class LayerwiseActivationExtractor(ActivationExtractor):
    """Production placeholder for extracting selected layer activations."""

    def extract(self, image_bytes: bytes) -> list[ActivationTensorInfo]:
        _ = image_bytes
        return [
            ActivationTensorInfo(
                layer_name="backbone.stage3.conv2",
                shape=[1, 512, 28, 28],
                summary_stats={"mean": 0.011, "std": 0.084, "max": 2.31},
            ),
            ActivationTensorInfo(
                layer_name="head.cls_logits",
                shape=[1, 80],
                summary_stats={"mean": 0.33, "std": 0.52, "max": 4.1},
            ),
        ]


class ForceDirectedNetwork3DVisualizer(Network3DVisualizer):
    """Production placeholder for exporting a 3D graph (e.g., glTF, Three.js JSON)."""

    def render(self, activations: list[ActivationTensorInfo]) -> NetworkState3D:
        node_count = len(activations) * 8
        edge_count = len(activations) * 12
        return NetworkState3D(
            format="gltf",
            uri="s3://neurovision-artifacts/network-state/request_id/network_graph.gltf",
            node_count=node_count,
            edge_count=edge_count,
        )
