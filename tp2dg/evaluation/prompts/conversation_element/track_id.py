"""
Track ID Evaluation.

This module evaluates recommendation quality and progress towards goal achievement,
assessing if the conversation is making progress by recommending relevant tracks.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns, get_recommended_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

track_id_recommendation_prompt = PromptTemplate(
    name="track_id_recommendation",
    version="v1.0",
    description="Evaluate the relevance of each RecSys recommendation to the listener's immediate request",
    required_params=[
        "conversation_goal",
        "conversation_turns",
        "recommended_tracks_content",
    ],
    response_expected_fields=[
        "recommendation_score",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Assess the RELEVANCE of each RecSys recommendation to the listener's immediate request in that turn.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## RECOMMENDED TRACKS:
{recommended_tracks_content}

## EVALUATION TASK:
Rate the RELEVANCE of each RecSys recommendation to the listener's immediate request in that specific turn. Focus ONLY on whether the recommended track matches what the listener asked for in their message.

**Scoring Criteria (1-4 scale):**
- **4**: Excellent Relevance - The recommended track is highly relevant to the listener's request
- **3**: Good Relevance - The recommended track is relevant to the listener's request
- **2**: Partial Relevance - The recommended track has some relevance but misses key aspects of the request
- **1**: Poor Relevance - The recommended track is not relevant to the listener's request

**Focus ONLY on:**
- **Request-Recommendation Alignment**: Does the recommended track match what the listener specifically asked for in their message?
- **Immediate Relevance**: Is the recommendation relevant to the listener's current request, not previous requests?

**Evaluation Method:**
For each turn, examine:
1. What did the listener specifically request in their message?
2. What track did the RecSys recommend?
3. How well does the recommendation match the immediate request?

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
recommendation_score: [1-4]
```
""",
)


class TrackIdEvaluator(BaseEvaluationTemplate):
    """Evaluates recommendation quality and progress towards goal achievement."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "track_id"
        self.prompt_template = track_id_recommendation_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the track ID evaluation prompt."""
        if uploaded_audio_files is None:
            uploaded_audio_files = {}
        if uploaded_image_files is None:
            uploaded_image_files = {}

        # Extract conversation goal
        conversation_goal = chat_json["conversation_goal"]
        goal_text = json.dumps(conversation_goal)

        # Extract conversation turns using utility function
        conversation_turns = extract_conversation_turns(chat_json)
        turns_text = json.dumps(conversation_turns)

        recommended_tracks_content = get_recommended_tracks_content(
            chat_json,
            uploaded_audio_files,
            uploaded_image_files,
        )

        return {
            "conversation_goal": goal_text,
            "conversation_turns": turns_text,
            "recommended_tracks_content": recommended_tracks_content,
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

        # Use empty dicts if no uploaded files provided (for backwards compatibility)
        if uploaded_audio_files is None:
            uploaded_audio_files = {}
        if uploaded_image_files is None:
            uploaded_image_files = {}

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

            # Extract just the score
            score = parsed["recommendation_score"]
            if isinstance(score, str):
                score = int(score)

            return {
                "recommendation_score": score,
                "success": score > 0 and score <= 4,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "recommendation_score": 0,
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
                "average_score": 0.0,
                "score_distribution": {},
                "success_rate": 0.0,
            }

        scores = [r["recommendation_score"] for r in successful_results]

        # Calculate distribution
        score_distribution = {i: scores.count(i) for i in range(1, 5)}

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "average_score": sum(scores) / len(scores),
            "score_distribution": score_distribution,
            "success_rate": len(successful_results) / len(individual_results),
            "scores": scores,  # For further analysis
        }


# Create the evaluator instance for easy import
track_id_evaluator = TrackIdEvaluator()
 