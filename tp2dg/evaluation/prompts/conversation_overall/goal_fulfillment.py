"""
Goal Fulfillment Evaluation.

This module evaluates whether the initial conversation goal is fulfilled or not,
classifying the conversation outcome as True (fulfilled) or False (not fulfilled).
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns, get_recommended_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

goal_fulfillment_prompt = PromptTemplate(
    name="goal_fulfillment",
    version="v1.0",
    description="Classify whether the initial conversation goal is fulfilled at the end of the conversation",
    required_params=[
        "conversation_goal",
        "conversation_turns",
        "recommended_tracks_content",
    ],
    response_expected_fields=[
        "goal_fulfilled",
    ],
    template="""
You are an expert evaluator of music recommendation systems. Assess whether the initial conversation goal has been fulfilled by the end of the conversation.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## RECOMMENDED TRACKS:
{recommended_tracks_content}

## EVALUATION TASK:
Classify whether the initial conversation goal is fulfilled or not by examining the FINAL OUTCOME of the conversation.

**Focus ONLY on:**
- **Goal Achievement**: Does the conversation end with the stated goal being achieved?
- **Final Outcome**: Can we definitively say the goal was fulfilled based on the conversation's conclusion?

**Classification Criteria:**
- **True (Fulfilled)**: The conversation ends with the stated goal being achieved
  - The goal stated in the conversation_goal has been accomplished
  - The conversation reaches a conclusion where the goal is met
  - There is clear evidence that the requested outcome was delivered

- **False (Not Fulfilled)**: The conversation ends without achieving the stated goal
  - The goal stated in the conversation_goal has NOT been accomplished
  - The conversation ends without delivering the requested outcome
  - The goal remains unfulfilled despite the conversation's conclusion

**Evaluation Method:**
1. Identify the specific goal from the conversation_goal
2. Examine the final state of the conversation
3. Determine if the goal was achieved by the end
4. Classify as fulfilled (true) or not fulfilled (false)

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
goal_fulfilled: true/false
```
""",
)


class GoalFulfillmentEvaluator(BaseEvaluationTemplate):
    """Evaluates whether the initial conversation goal is fulfilled."""

    def __init__(self):
        super().__init__()
        self.evaluation_name = "goal_fulfillment"
        self.prompt_template = goal_fulfillment_prompt

    def prepare_prompt_data(
        self,
        chat_json: dict[str, Any],
        uploaded_audio_files: Optional[dict[str, Any]] = None,
        uploaded_image_files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Prepare data for the goal fulfillment prompt."""

        # Use empty dicts if no uploaded files provided (for backwards compatibility)
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

            # Extract goal fulfillment
            goal_fulfilled = parsed["goal_fulfilled"]

            # Convert to boolean if needed
            if isinstance(goal_fulfilled, str):
                goal_fulfilled = goal_fulfilled.lower() == "true"

            return {
                "goal_fulfilled": goal_fulfilled,
                "success": True,
                "raw_response": response,
                "parsed_response": parsed,
            }

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {
                "goal_fulfilled": False,
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
                "goal_fulfillment_rate": 0.0,
                "fulfilled_count": 0,
                "not_fulfilled_count": 0,
            }

        # Extract goal fulfillment results
        fulfilled_results = [r for r in successful_results if r.get("goal_fulfilled", False)]
        not_fulfilled_results = [r for r in successful_results if not r.get("goal_fulfilled", False)]

        fulfillment_rate = len(fulfilled_results) / len(successful_results)

        return {
            "total_conversations": len(individual_results),
            "successful_evaluations": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "goal_fulfillment_rate": fulfillment_rate,
            "fulfilled_count": len(fulfilled_results),
            "not_fulfilled_count": len(not_fulfilled_results),
            "fulfilled_conversations": [r.get("conversation_id", "unknown") for r in fulfilled_results],
            "not_fulfilled_conversations": [r.get("conversation_id", "unknown") for r in not_fulfilled_results],
        }


# Create the evaluator instance for easy import
goal_fulfillment_evaluator = GoalFulfillmentEvaluator()
