from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class SemanticInsight:
    scene_description: str
    tags: list[str]
    reasoning: str


@dataclass(slots=True)
class HeatmapArtifact:
    algorithm: str
    uri: str
    overlay_uri: str


@dataclass(slots=True)
class ActivationTensorInfo:
    layer_name: str
    shape: list[int]
    summary_stats: dict[str, float]


@dataclass(slots=True)
class NetworkState3D:
    format: str
    uri: str
    node_count: int
    edge_count: int


@dataclass(slots=True)
class PipelineMetadata:
    request_id: str
    model_version: str
    elapsed_ms: float
    created_at: str

    @classmethod
    def create(cls, request_id: str, model_version: str, elapsed_ms: float) -> "PipelineMetadata":
        timestamp = datetime.now(timezone.utc).isoformat()
        return cls(
            request_id=request_id,
            model_version=model_version,
            elapsed_ms=elapsed_ms,
            created_at=timestamp,
        )


@dataclass(slots=True)
class NeuroVisionResponse:
    metadata: PipelineMetadata
    detections: list[Detection]
    semantic: SemanticInsight
    explainability_heatmap: HeatmapArtifact
    activations: list[ActivationTensorInfo]
    network_state_3d: NetworkState3D

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
