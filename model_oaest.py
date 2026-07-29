"""Backward-compatible model import.

The old repository instantiated ``SoftDP(horizon=1)`` inside this file.  The
implementation now delegates to the sequence model in :mod:`novanet.model`,
which requires an explicit finite horizon and future candidate sequence.
"""

from novanet.model import NovaNet

PCGNN_OAEST = NovaNet

__all__ = ["NovaNet", "PCGNN_OAEST"]
