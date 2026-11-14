"""Synthetic customer-support dialogues with ground-truth labels.

Stages: dice (hidden parameters) -> scenario -> beat sheet -> draft -> stylize (routed
through cached micro-arena winners) -> macro judge (QA gate with retries).
"""

from judgelab.dialogues.pipeline import Dialogue, PipelineModels, run_pipeline

__all__ = ["Dialogue", "PipelineModels", "run_pipeline"]
