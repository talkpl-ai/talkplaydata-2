"""
Conversation Goal Alignment and Distribution Evaluation.

This module evaluates the alignment between the intended conversation goal and the actual conversation,
classifying conversations into specificity (HH/HL/LH/LL) and category (A/B/C/D/E/F/G/H/I/J/K) classes.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns
from tp2dg.prompts.prompt_template import PromptTemplate

conversation_goal_alignment_prompt = PromptTemplate(
    name="conversation_goal_alignment",
    version="v1.0",
    description="Classify conversation into specificity and category classes based on goal alignment",
    required_params=[
        "conversation_goal",
        "conversation_turns",
    ],
    response_expected_fields=[
        "specificity_class",
        "category_class",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Classify the conversation into specificity and category classes based on the conversation goal and actual conversation flow.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## EVALUATION TASK:
Classify the conversation into two dimensions based on the conversation goal and actual conversation flow.

### 1. Specificity Classification: `specificity_class`
Classify the conversation specificity based on goal clarity and conversation precision:

**HH (High-High)**: High goal specificity with high conversation precision
- Clear, specific goal (e.g., "find the exact album 'Dark Side of the Moon' by Pink Floyd")
- Conversation shows precise, targeted recommendations
- High accuracy in matching goal requirements

**HL (High-Low)**: High goal specificity with low conversation precision
- Clear, specific goal but conversation lacks precision
- Goal is well-defined but recommendations are broad or imprecise
- Specific request but general response quality

**LH (Low-High)**: Low goal specificity with high conversation precision
- Broad, general goal but conversation shows high precision
- Vague goal but very targeted, specific recommendations
- General request but precise response quality

**LL (Low-Low)**: Low goal specificity with low conversation precision
- Broad, general goal with broad, general conversation
- Vague goal with imprecise, general recommendations
- Both goal and conversation lack specificity

### 2. Category Classification: `category_class`
Classify the conversation category based on the type of music recommendation goal:

**A**: Album Discovery - Finding specific albums or albums by specific artists
**B**: Genre Exploration - Exploring music within specific genres
**C**: Mood-Based Recommendations - Finding music for specific moods or activities
**D**: Artist Discovery - Discovering new artists or exploring artist catalogs
**E**: Era/Decade Exploration - Finding music from specific time periods
**F**: Cultural/Regional Music - Exploring music from specific cultures or regions
**G**: Collaborative Filtering - Recommendations based on similar user preferences
**H**: Contextual Recommendations - Music for specific situations or contexts
**I**: Technical/Production Quality - Focus on audio quality, production, or technical aspects
**J**: Cross-Genre Exploration - Mixing or transitioning between different genres
**K**: Personal Collection Management - Organizing, curating, or managing personal music

**Consider for Classification:**
- Goal Content: What is the primary focus of the conversation goal?
- Conversation Flow: How does the actual conversation align with the intended goal?
- Recommendation Patterns: What types of recommendations are being made?
- User Intent: What is the underlying user intention in the conversation?

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
specificity_class: "HH"/"HL"/"LH"/"LL"
category_class: "A"/"B"/"C"/"D"/"E"/"F"/"G"/"H"/"I"/"J"/"K"
```
""",
)


class ConversationGoalAlignmentEvaluator(BaseEvaluationTemplate):
    """Evaluates conversation goal alignment and classifies into specificity and category classes."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "conversation_goal_alignment"
        self.prompt_template = conversation_goal_alignment_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the conversation goal alignment prompt."""

        # Extract conversation goal
        conversation_goal = chat_json["conversation_goal"]
        goal_text = json.dumps(conversation_goal)

        # Extract conversation turns using utility function
        conversation_turns = extract_conversation_turns(chat_json)
        turns_text = json.dumps(conversation_turns)

        return {
            "conversation_goal": goal_text,
            "conversation_turns": turns_text,
        }

    async def evaluate_single(
        self,
        conversation_data: dict[str, Any],
        llm_call_func,
        client,
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Evaluate a single conversation using centralized uploaded files."""

        # Get prompt data with uploaded files
        prompt_data = self.prepare_prompt_data(conversation_data, uploaded_audio_files, uploaded_image_files)

        content = self.prompt_template.format(**prompt_data)

        # Call LLM
        response = await llm_call_func(content, client)

        # Parse response
        try:
            parsed = robust_parse_yaml_response(
                response,
                self.prompt_template.response_expected_fields,
            )

            # Extract classifications
            specificity_class = parsed["specificity_class"]
            category_class = parsed["category_class"]

            # Validate specificity class
            valid_specificities = ["HH", "HL", "LH", "LL"]
            if specificity_class not in valid_specificities:
                raise ValueError(f"Invalid specificity class: {specificity_class}")

            # Validate category class
            valid_categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
            if category_class not in valid_categories:
                raise ValueError(f"Invalid category class: {category_class}")

            return {
                "specificity_class": specificity_class,
                "category_class": category_class,
                "success": True,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "specificity_class": "Unknown",
                "category_class": "Unknown",
                "success": False,
                "raw_response": response,
                "error": str(e),
            }

    def aggregate_results(self, individual_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate individual results into summary statistics."""

        successful_results = [r for r in individual_results if r.get("success", False)]

        if not successful_results:
            return {
                "total_conversations": len(individual_results),
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "specificity_distribution": {},
                "category_distribution": {},
                "combined_distribution": {},
            }

        # Extract classifications
        specificity_classes = [r.get("specificity_class", "LL") for r in successful_results]
        category_classes = [r.get("category_class", "A") for r in successful_results]

        # Calculate distributions
        specificity_dist = {}
        for spec in ["HH", "HL", "LH", "LL"]:
            count = specificity_classes.count(spec)
            specificity_dist[spec] = {
                "count": count,
                "percentage": count / len(successful_results) * 100,
            }

        category_dist = {}
        for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
            count = category_classes.count(cat)
            category_dist[cat] = {
                "count": count,
                "percentage": count / len(successful_results) * 100,
            }

        # Calculate combined distribution
        combined_dist = {}
        for spec in ["HH", "HL", "LH", "LL"]:
            for cat in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]:
                key = f"{spec}-{cat}"
                count = sum(
                    1
                    for r in successful_results
                    if r.get("specificity_class") == spec and r.get("category_class") == cat
                )
                combined_dist[key] = {
                    "count": count,
                    "percentage": count / len(successful_results) * 100,
                }

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "specificity_distribution": specificity_dist,
            "category_distribution": category_dist,
            "combined_distribution": combined_dist,
            "individual_results": [
                {
                    "conversation_id": r.get("conversation_id", "unknown"),
                    "specificity_class": r.get("specificity_class"),
                    "category_class": r.get("category_class"),
                }
                for r in successful_results
            ],
        }


# Create the evaluator instance for easy import
conversation_goal_alignment_evaluator = ConversationGoalAlignmentEvaluator()
