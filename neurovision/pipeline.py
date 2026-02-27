from __future__ import annotations

import time
from dataclasses import dataclass

from neurovision.interfaces import (
    ActivationExtractor,
    ExplainabilityEngine,
    Network3DVisualizer,
    ObjectDetector,
    SemanticAnalyzer,
)
from neurovision.models import NeuroVisionResponse, PipelineMetadata


@dataclass(slots=True)
class NeuroVisionPipeline:
    detector: ObjectDetector
    semantic_analyzer: SemanticAnalyzer
    explainability_engine: ExplainabilityEngine
    activation_extractor: ActivationExtractor
    visualizer_3d: Network3DVisualizer
    model_version: str = "neurovision-v1"

    def run(self, image_bytes: bytes, request_id: str) -> NeuroVisionResponse:
        start = time.perf_counter()

        detections = self.detector.detect(image_bytes)
        semantic = self.semantic_analyzer.analyze(image_bytes, detections)
        heatmap = self.explainability_engine.generate(image_bytes, detections)
        activations = self.activation_extractor.extract(image_bytes)
        network_state = self.visualizer_3d.render(activations)

        elapsed_ms = (time.perf_counter() - start) * 1000
        metadata = PipelineMetadata.create(
            request_id=request_id,
            model_version=self.model_version,
            elapsed_ms=round(elapsed_ms, 2),
        )
        return NeuroVisionResponse(
            metadata=metadata,
            detections=detections,
            semantic=semantic,
            explainability_heatmap=heatmap,
            activations=activations,
            network_state_3d=network_state,
        )
