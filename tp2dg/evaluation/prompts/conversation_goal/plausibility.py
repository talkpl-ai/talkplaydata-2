"""
Conversation Goal Plausibility Evaluation.

This module evaluates whether conversation goals are plausible/achievable
given the recommendation pool, using the same Track entities and file upload
system as the data generation process.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import get_recommended_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

conversation_goal_plausibility_prompt = PromptTemplate(
	name="conversation_goal_plausibility",
	version="v1.0",
	description="Evaluate whether conversation goals are plausible/achievable given the recommendation pool",
	required_params=[
		"conversation_goal",
		"recommendation_pool_content",
	],
	response_expected_fields=[
		"plausibility_score",
	],
	template="""
You are an expert evaluator of music recommendation systems. Assess whether the given conversation goal is plausible and achievable given the available recommendation pool.

## CONVERSATION GOAL:
{conversation_goal}

## RECOMMENDATION POOL (Available tracks to recommend):
{recommendation_pool_content}

## EVALUATION TASK:
Rate the plausibility of achieving this conversation goal given the recommendation pool and listener profile.

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Goal is highly realistic and perfectly matched to pool
- **3**: Good - Goal is realistic and achievable with available tracks
- **2**: Fair - Goal is achievable but may require compromises
- **1**: Poor - Goal is unrealistic or impossible given the available recommendation pool

**Consider:**
- Goal-Pool Alignment: Does the recommendation pool contain sufficient matching tracks? Is the goal achievable with the available tracks by conversation?

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
plausibility_score: [1-4]
```
""",
)


class ConversationGoalPlausibilityEvaluator(BaseEvaluationTemplate):
	"""Evaluates conversation goal plausibility using proper Track entities and centralized file uploads."""

	def __init__(self):
		super().__init__()
		self.evaluation_name = "conversation_goal_plausibility"
		self.prompt_template = conversation_goal_plausibility_prompt

	def prepare_prompt_data(
		self,
		chat_json: dict[str, Any],
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""Prepare data for the conversation goal plausibility prompt."""

		# Extract conversation goal
		conversation_goal = chat_json["conversation_goal"]
		goal_text = json.dumps(conversation_goal, indent=2)

		recommendation_pool_content = get_recommended_tracks_content(
			chat_json,
			uploaded_audio_files,
			uploaded_image_files,
		)

		return {
			"conversation_goal": goal_text,
			"recommendation_pool_content": recommendation_pool_content,
		}

	async def evaluate_single(
		self,
		chat_json: dict[str, Any],
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
		prompt_data = self.prepare_prompt_data(chat_json, uploaded_audio_files, uploaded_image_files)

		# Create the prompt
		content = self.prompt_template.format(**prompt_data)

		# Call LLM
		response = await llm_call_func(content, client)

		# Parse response
		try:
			parsed = robust_parse_yaml_response(
				response,
				self.prompt_template.response_expected_fields,
			)

			# Extract just the score (simplified as requested)
			score = parsed["plausibility_score"]
			if isinstance(score, str):
				score = int(score)

			return {
				"plausibility_score": score,
				"success": score > 0 and score <= 4,
				"raw_response": response,
				"parsed_response": parsed,
			}

		except Exception as e:
			logging.error(f"Error parsing response: {e}")
			return {
				"plausibility_score": 0,
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

		scores = [r["plausibility_score"] for r in successful_results]

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
conversation_goal_plausibility_evaluator = ConversationGoalPlausibilityEvaluator()
 