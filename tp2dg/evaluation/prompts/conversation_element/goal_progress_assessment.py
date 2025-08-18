"""
Goal Progress Assessment Evaluation.

This module evaluates whether the goal_progress_assessment is correctly labeled
given the conversation goal and actual progress made.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import (
	extract_conversation_turns,
	extract_goal_progress_assessments,
	get_recommended_tracks_content,
)
from tp2dg.prompts.prompt_template import PromptTemplate

goal_progress_assessment_prompt = PromptTemplate(
	name="goal_progress_assessment",
	version="v1.0",
	description="Evaluate whether the goal_progress_assessment is correctly labeled",
	required_params=[
		"conversation_goal",
		"conversation_turns",
		"recommended_tracks_content",
		"goal_progress_assessment",
	],
	response_expected_fields=[
		"accuracy_score",
	],
	# TODO: Update this prompt to take the whole conversation better, and then evaluate it.
	template="""
You are an expert evaluator of music recommendation systems. Assess whether the goal_progress_assessment is correctly labeled given the conversation goal and actual progress made.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## RECOMMENDED TRACKS:
{recommended_tracks_content}

## GOAL PROGRESS ASSESSMENT:
{goal_progress_assessment}

## EVALUATION TASK:
Rate the accuracy of the goal_progress_assessment label given the conversation goal and actual progress.

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Assessment perfectly reflects the actual progress towards the goal
- **3**: Good - Assessment is mostly accurate with minor discrepancies
- **2**: Fair - Assessment has some accuracy issues but captures main progress
- **1**: Poor - Assessment is inaccurate or misrepresents the actual progress

**Consider:**
- Goal Alignment: Does the assessment correctly reflect progress towards the stated goal?
- Turn Analysis: Does the assessment match what actually happened in the conversation?
- Track Relevance: Are the recommended tracks relevant to the goal progress assessment?
- Consistency: Is the assessment consistent with the conversation flow?

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
accuracy_score: [1-4]
```
""",
)


class GoalProgressAssessmentEvaluator(BaseEvaluationTemplate):
	"""Evaluates goal progress assessment accuracy using LLM-as-a-judge."""

	def __init__(self):
		super().__init__()
		self.evaluation_name = "goal_progress_assessment"
		self.prompt_template = goal_progress_assessment_prompt

	def prepare_prompt_data(
		self,
		chat_json: dict[str, Any],
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""Prepare data for the goal progress assessment prompt."""

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

		# Extract goal progress assessments using utility function
		goal_progress_assessments = extract_goal_progress_assessments(chat_json)
		assessment_text = json.dumps(goal_progress_assessments)

		return {
			"conversation_goal": goal_text,
			"conversation_turns": turns_text,
			"recommended_tracks_content": recommended_tracks_content,
			"goal_progress_assessment": assessment_text,
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
			score = parsed["accuracy_score"]
			if isinstance(score, str):
				score = int(score)

			return {
				"accuracy_score": score,
				"success": score > 0 and score <= 4,
				"raw_response": response,
				"parsed_response": parsed,
			}

		except Exception as e:
			logging.error(f"Error parsing response: {e}")
			return {
				"accuracy_score": 0,
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

		scores = [r["accuracy_score"] for r in successful_results]

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
goal_progress_assessment_evaluator = GoalProgressAssessmentEvaluator()
 