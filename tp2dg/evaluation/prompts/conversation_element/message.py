"""
Message Evaluation.

This module evaluates message quality for both Listener and Recsys,
including naturalness, realism, consistency, and performance aspects.
"""

import json
import logging
from typing import Any, Optional

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate
from tp2dg.evaluation.prompts.utils import extract_conversation_turns, get_recommended_tracks_content
from tp2dg.prompts.prompt_template import PromptTemplate

message_evaluation_prompt = PromptTemplate(
	name="message_evaluation",
	version="v1.0",
	description="Evaluate message quality for both Listener and Recsys in a single call",
	required_params=[
		"conversation_goal",
		"conversation_turns",
		"listener_profile",
		"recommended_tracks_content",
	],
	response_expected_fields=[
		"listener_quality_score",
		"recsys_quality_score",
		"listener_helpfulness_score",
		"recsys_accuracy_score",
	],
	template="""
You are an expert evaluator of music recommendation systems. Assess message quality for both Listener and Recsys in multiple dimensions.

## CONVERSATION GOAL:
{conversation_goal}

## CONVERSATION TURNS:
{conversation_turns}

## LISTENER PROFILE:
{listener_profile}

## RECOMMENDED TRACKS:
{recommended_tracks_content}

## EVALUATION TASKS:

### 1. Listener Message Quality (Naturalness, Realism, Consistency): `listener_quality_score`
Rate how realistic, natural, and consistent the Listener messages are.

**Consider:**
- Naturalness: Do the messages sound like natural human conversation? Do they use appropriate language and tone?
- Realism: Are the messages realistic given the listener's profile (age, musical preferences, cultural background)?
- Consistency: Are the messages consistent with the listener's stated preferences and behavior throughout the conversation?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Messages are highly natural, realistic, and consistent with the profile
- **3**: Good - Messages are mostly natural and realistic with minor inconsistencies
- **2**: Fair - Messages have some naturalness issues or inconsistencies
- **1**: Poor - Messages are unnatural, unrealistic, or inconsistent

### 2. Recsys Message Quality (Naturalness, Realism, Consistency): `recsys_quality_score`
Rate how realistic, natural, and consistent the Recsys messages are.

**Consider:**
- Naturalness: Do the messages sound like natural conversation from a recommendation system? Are they conversational and engaging?
- Realism: Are the messages realistic given the recommendation context and available tracks?
- Consistency: Are the messages consistent with the conversation goal and previous interactions?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Messages are highly natural, realistic, and consistent with recommendation context
- **3**: Good - Messages are mostly natural and realistic with minor inconsistencies
- **2**: Fair - Messages have some naturalness issues or inconsistencies
- **1**: Poor - Messages are unnatural, unrealistic, or inconsistent

### 3. Listener Query Helpfulness: `listener_helpfulness_score`
Rate how helpful the Listener's queries are in steering the conversation towards achieving the goal.

**Consider:**
- Goal Alignment: Do the queries help move towards the stated conversation goal?
- Clarity: Are the queries clear and actionable for the recommendation system?
- Progress: Do the queries contribute to making progress in the conversation?
- Specificity: Are the queries specific enough to guide recommendations effectively?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Queries effectively guide the conversation towards goal achievement
- **3**: Good - Queries are mostly helpful with minor inefficiencies
- **2**: Fair - Queries have some helpfulness but could be more effective
- **1**: Poor - Queries are unhelpful or counterproductive to goal achievement

### 4. Recsys Response Accuracy: `recsys_accuracy_score`
Rate whether the Recsys responses are factual and correctly describe the recommended tracks.

**Consider:**
- Factual Accuracy: Are the track descriptions and information accurate?
- No Hallucination: Does the system avoid making up information about tracks?
- Consistency: Are the responses consistent with the actual track data?
- Relevance: Do the responses accurately describe the specific tracks being recommended?

**Scoring Criteria (1-4 scale):**
- **4**: Excellent - Responses are completely factual and accurate
- **3**: Good - Responses are mostly factual with minor inaccuracies
- **2**: Fair - Responses have some factual issues but are generally correct
- **1**: Poor - Responses contain significant factual errors or hallucinations

## RESPONSE FORMAT:
Respond with ONLY this YAML format:

```yaml
listener_quality_score: [1-4]
recsys_quality_score: [1-4]
listener_helpfulness_score: [1-4]
recsys_accuracy_score: [1-4]
```
""",
)


class MessageEvaluator(BaseEvaluationTemplate):
	"""Evaluates message quality for both Listener and Recsys in a single call."""

	def __init__(self):
		super().__init__()
		self.evaluation_name = "message"
		self.prompt_template = message_evaluation_prompt

	def prepare_prompt_data(
		self,
		chat_json: dict[str, Any],
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""Prepare data for the message evaluation prompt."""

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

		# Extract listener profile
		listener_profile = chat_json["listener_profile"]
		profile_text = json.dumps(listener_profile)

		recommended_tracks_content = get_recommended_tracks_content(
			chat_json,
			uploaded_audio_files,
			uploaded_image_files,
		)

		return {
			"conversation_goal": goal_text,
			"conversation_turns": turns_text,
			"listener_profile": profile_text,
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

			# Extract all scores
			listener_quality_score = parsed["listener_quality_score"]
			recsys_quality_score = parsed["recsys_quality_score"]
			listener_helpfulness_score = parsed["listener_helpfulness_score"]
			recsys_accuracy_score = parsed["recsys_accuracy_score"]

			# Convert to int if needed
			if isinstance(listener_quality_score, str):
				listener_quality_score = int(listener_quality_score)
			if isinstance(recsys_quality_score, str):
				recsys_quality_score = int(recsys_quality_score)
			if isinstance(listener_helpfulness_score, str):
				listener_helpfulness_score = int(listener_helpfulness_score)
			if isinstance(recsys_accuracy_score, str):
				recsys_accuracy_score = int(recsys_accuracy_score)

			# Check if all scores are valid
			all_scores_valid = all(
				0 < score <= 4
				for score in [
					listener_quality_score,
					recsys_quality_score,
					listener_helpfulness_score,
					recsys_accuracy_score,
				]
			)

			return {
				"listener_quality_score": listener_quality_score,
				"recsys_quality_score": recsys_quality_score,
				"listener_helpfulness_score": listener_helpfulness_score,
				"recsys_accuracy_score": recsys_accuracy_score,
				"success": all_scores_valid,
				"raw_response": response,
				"parsed_response": parsed,
			}

		except Exception as e:
			logging.error(f"Error parsing response: {e}")
			return {
				"listener_quality_score": 0,
				"recsys_quality_score": 0,
				"listener_helpfulness_score": 0,
				"recsys_accuracy_score": 0,
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
				"metrics": {
					"listener_quality": {"average_score": 0.0, "score_distribution": {}},
					"recsys_quality": {"average_score": 0.0, "score_distribution": {}},
					"listener_helpfulness": {"average_score": 0.0, "score_distribution": {}},
					"recsys_accuracy": {"average_score": 0.0, "score_distribution": {}},
				},
			}

		# Extract all scores
		listener_quality_scores = [r["listener_quality_score"] for r in successful_results]
		recsys_quality_scores = [r["recsys_quality_score"] for r in successful_results]
		listener_helpfulness_scores = [r["listener_helpfulness_score"] for r in successful_results]
		recsys_accuracy_scores = [r["recsys_accuracy_score"] for r in successful_results]

		# Calculate distributions
		listener_quality_dist = {i: listener_quality_scores.count(i) for i in range(1, 5)}
		recsys_quality_dist = {i: recsys_quality_scores.count(i) for i in range(1, 5)}
		listener_helpfulness_dist = {i: listener_helpfulness_scores.count(i) for i in range(1, 5)}
		recsys_accuracy_dist = {i: recsys_accuracy_scores.count(i) for i in range(1, 5)}

		return {
			"total_conversations": len(individual_results),
			"successful_evaluations": len(successful_results),
			"success_rate": len(successful_results) / len(individual_results),
			"metrics": {
				"listener_quality": {
					"average_score": sum(listener_quality_scores) / len(listener_quality_scores),
					"score_distribution": listener_quality_dist,
					"scores": listener_quality_scores,
				},
				"recsys_quality": {
					"average_score": sum(recsys_quality_scores) / len(recsys_quality_scores),
					"score_distribution": recsys_quality_dist,
					"scores": recsys_quality_scores,
				},
				"listener_helpfulness": {
					"average_score": sum(listener_helpfulness_scores) / len(listener_helpfulness_scores),
					"score_distribution": listener_helpfulness_dist,
					"scores": listener_helpfulness_scores,
				},
				"recsys_accuracy": {
					"average_score": sum(recsys_accuracy_scores) / len(recsys_accuracy_scores),
					"score_distribution": recsys_accuracy_dist,
					"scores": recsys_accuracy_scores,
				},
			},
		}


# Create the evaluator instance for easy import
message_evaluator = MessageEvaluator()
