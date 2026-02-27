from neurovision.components import (
    BaselineObjectDetector,
    BaselineSemanticAnalyzer,
    ForceDirectedNetwork3DVisualizer,
    GradCamExplainabilityEngine,
    LayerwiseActivationExtractor,
)
from neurovision.pipeline import NeuroVisionPipeline


def build_default_pipeline() -> NeuroVisionPipeline:
    return NeuroVisionPipeline(
        detector=BaselineObjectDetector(),
        semantic_analyzer=BaselineSemanticAnalyzer(),
        explainability_engine=GradCamExplainabilityEngine(),
        activation_extractor=LayerwiseActivationExtractor(),
        visualizer_3d=ForceDirectedNetwork3DVisualizer(),
    )
