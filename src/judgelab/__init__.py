"""judgelab: experiments on LLM-as-judge reliability with local models.

Two parts share one LLM backend:

* :mod:`judgelab.cross_judge` -- every model writes prompts, generates and judges every
  other model's output, so judge bias can be separated from answer quality.
* :mod:`judgelab.dialogues` -- a staged synthetic-dialogue generator whose stylization
  step picks a (prompter, actor) pair per context with a small judged tournament and
  caches the winner.
"""

__version__ = "0.1.0"
