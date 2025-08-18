"""
Conversation Overall Evaluation Package.

This package contains evaluators for overall conversation quality assessment:
- Goal Fulfillment: Classifies if the initial goal is fulfilled (True/False)
- Conversation Goal Alignment: Classifies conversations into specificity and category classes
- Multimodality: Rates multimodal aspect consideration (True/False/NotRelevant)
"""

from .conversation_goal_alignment_distribution import conversation_goal_alignment_evaluator
from .goal_fulfillment import goal_fulfillment_evaluator
from .multimodality import multimodality_evaluator

__all__ = [
    "conversation_goal_alignment_evaluator",
    "goal_fulfillment_evaluator",
    "multimodality_evaluator",
]
