from __future__ import annotations

from typing import Protocol

from neurovision.models import (
    ActivationTensorInfo,
    Detection,
    HeatmapArtifact,
    NetworkState3D,
    SemanticInsight,
)


class ObjectDetector(Protocol):
    def detect(self, image_bytes: bytes) -> list[Detection]:
        """Return detected objects for an input image."""


class SemanticAnalyzer(Protocol):
    def analyze(self, image_bytes: bytes, detections: list[Detection]) -> SemanticInsight:
        """Generate high-level scene understanding."""


class ExplainabilityEngine(Protocol):
    def generate(self, image_bytes: bytes, detections: list[Detection]) -> HeatmapArtifact:
        """Generate an explainability heatmap and overlays."""


class ActivationExtractor(Protocol):
    def extract(self, image_bytes: bytes) -> list[ActivationTensorInfo]:
        """Extract relevant layer activations from the network."""


class Network3DVisualizer(Protocol):
    def render(self, activations: list[ActivationTensorInfo]) -> NetworkState3D:
        """Build a 3D graph representation of model internals."""
