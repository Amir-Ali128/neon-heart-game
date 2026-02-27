# NEUROVISION AI - Full Pipeline Specification

## 1) Mission
NEUROVISION AI is a modular computer vision pipeline that accepts an image, performs object detection, semantic interpretation, explainability generation, activation extraction, and 3D internal-state visualization, then returns a structured JSON response suitable for downstream services.

## 2) End-to-End Dataflow
1. **Ingress**: API receives `multipart/form-data` with an image.
2. **Object Detection**: Detector predicts classes, confidence, and bounding boxes.
3. **Semantic Understanding**: Semantic stage summarizes scene meaning and tags.
4. **Explainability Heatmaps**: Explainability stage outputs model attention artifacts.
5. **Neural Activations**: Activation extractor captures selected layer tensors.
6. **3D Network State**: Visualizer exports a graph-like 3D network representation.
7. **Structured Response**: Service emits typed JSON for clients and observability.

## 3) Module Boundaries
- `ObjectDetector`: `detect(image_bytes) -> list[Detection]`
- `SemanticAnalyzer`: `analyze(image_bytes, detections) -> SemanticInsight`
- `ExplainabilityEngine`: `generate(image_bytes, detections) -> HeatmapArtifact`
- `ActivationExtractor`: `extract(image_bytes) -> list[ActivationTensorInfo]`
- `Network3DVisualizer`: `render(activations) -> NetworkState3D`
- `NeuroVisionPipeline`: orchestrates stage execution and metadata generation

This isolates each capability to enable independent replacement (for example swapping YOLO with DETR, or Grad-CAM with Score-CAM) without changing the API contract.

## 4) API Contract
### Endpoint
`POST /api/neurovision/run`

### Input
- Content-Type: `multipart/form-data`
- Form field: `image`
- Optional header: `X-Request-Id`

### Success Response (200)
```json
{
  "metadata": {
    "request_id": "ea9f...",
    "model_version": "neurovision-v1",
    "elapsed_ms": 23.15,
    "created_at": "2026-02-27T12:00:00.000000+00:00"
  },
  "detections": [
    {
      "label": "person",
      "confidence": 0.97,
      "bbox": {"x_min": 0.14, "y_min": 0.12, "x_max": 0.67, "y_max": 0.95}
    }
  ],
  "semantic": {
    "scene_description": "A person is centered in the frame with high confidence detection.",
    "tags": ["person"],
    "reasoning": "Semantic context derived from object classes and spatial priors."
  },
  "explainability_heatmap": {
    "algorithm": "grad_cam",
    "uri": "s3://.../base_heatmap.npy",
    "overlay_uri": "s3://.../overlay.png"
  },
  "activations": [
    {
      "layer_name": "backbone.stage3.conv2",
      "shape": [1, 512, 28, 28],
      "summary_stats": {"mean": 0.011, "std": 0.084, "max": 2.31}
    }
  ],
  "network_state_3d": {
    "format": "gltf",
    "uri": "s3://.../network_graph.gltf",
    "node_count": 16,
    "edge_count": 24
  }
}
```

### Error Responses
- `400`: Missing or empty image
- `500`: Unhandled stage/runtime failure (recommended to map to structured error object in production)

## 5) Production-Readiness Considerations
- **Scalability**: Containerize and deploy with horizontal auto-scaling for CPU/GPU workers.
- **Asynchronous Mode**: Add job queue for large models and long explainability/3D rendering tasks.
- **Observability**: Emit request IDs, stage timings, and per-stage errors; integrate tracing.
- **Artifact Storage**: Persist heatmaps and 3D outputs in object storage (S3/GCS/Azure Blob).
- **Model Registry**: Track model versions and perform canary releases.
- **Security**: Validate file types/size; perform malware scanning for uploads.
- **Reliability**: Add circuit breakers and per-stage fallbacks when optional stages fail.
- **Compliance**: Redact sensitive data and apply retention controls for uploaded images.

## 6) Extension Points
- Replace baseline components with concrete production implementations:
  - Detector: YOLOv8 / RT-DETR / Detectron2
  - Semantic: CLIP + LLM reasoning or VLM caption model
  - Explainability: Grad-CAM++, Score-CAM, Integrated Gradients
  - Activation extraction: configurable layer selectors
  - 3D state: graph export for Three.js, Unity, or WebGL dashboards

## 7) Repository Mapping
- `neurovision/interfaces.py`: module interfaces
- `neurovision/components.py`: baseline stage implementations
- `neurovision/pipeline.py`: orchestrator
- `neurovision/models.py`: response schema
- `app.py`: HTTP ingress and pipeline invocation
