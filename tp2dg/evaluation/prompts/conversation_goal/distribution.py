"""
Conversation Goal Distribution Analysis.

This module analyzes the distribution of conversation goals by computing statistics
of specificities and categories across all conversations (no LLM needed).
"""

from collections import Counter
from typing import Any, Optional

from tp2dg.evaluation.prompts.eval_template import BaseEvaluationTemplate


class ConversationGoalDistributionEvaluator(BaseEvaluationTemplate):
	"""Evaluates conversation goal distribution by computing specificity and category statistics."""

	def __init__(self):
		super().__init__()
		self.evaluation_name = "conversation_goal_distribution"
		# No prompt template needed for computational evaluation
		self.prompt_template = None

	def prepare_prompt_data(
		self,
		chat_json: dict[str, Any],
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""
		Prepare data for the conversation goal distribution analysis.

		For computational evaluations, this just extracts the goal data.
		"""
		conversation_goal = chat_json["conversation_goal"]

		return {
			"conversation_goal": conversation_goal,
			"specificity": conversation_goal["specificity"],
			"category": conversation_goal["category"],
		}

	async def evaluate_single(
		self,
		chat_json: dict[str, Any],
		llm_call_func,
		client,
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""
		Evaluate a single conversation by extracting goal specificity and category.

		This is computational (no LLM call needed).
		"""

		# Extract conversation goal data
		conversation_goal = chat_json["conversation_goal"]

		specificity = conversation_goal["specificity"]  # "LL"
		category = conversation_goal["category"]  # "B"

		# Additional goal information for analysis
		goal_text = conversation_goal["listener_goal"]
		target_turns = conversation_goal["target_turn_count"]

		return {
			"specificity": specificity,
			"category": category,
			"goal_text": goal_text,
			"target_turns": target_turns,
			"success": True,
			"both_combined": f"{specificity}_{category}",  # For combined analysis
		}

	def aggregate_results(self, individual_results: list[dict[str, Any]]) -> dict[str, Any]:
		"""
		Aggregate individual results into distribution statistics and histograms.

		As specified in PLAN.md: "Compute the average over Specificity, Category, and Both (combined). Draw histogram."
		"""

		successful_results = [r for r in individual_results if r.get("success", False)]

		if not successful_results:
			return {
				"total_conversations": len(individual_results),
				"successful_evaluations": 0,
				"success_rate": 0.0,
				"specificity_distribution": {},
				"category_distribution": {},
				"combined_distribution": {},
				"target_turns_stats": {},
			}

		# Extract data for analysis
		specificities = [r["specificity"] for r in successful_results]
		categories = [r["category"] for r in successful_results]
		combined_specs_cats = [r["both_combined"] for r in successful_results]
		target_turns = [
			r["target_turns"]
			for r in successful_results
			if isinstance(r["target_turns"], (int, float)) and r["target_turns"] > 0
		]

		# Compute distributions (counts and percentages)
		specificity_counts = Counter(specificities)
		category_counts = Counter(categories)
		combined_counts = Counter(combined_specs_cats)

		total_count = len(successful_results)

		# Convert to percentage distributions for histogram data
		specificity_distribution = {
			spec: {
				"count": count,
				"percentage": (count / total_count) * 100,
			}
			for spec, count in specificity_counts.items()
		}

		category_distribution = {
			cat: {
				"count": count,
				"percentage": (count / total_count) * 100,
			}
			for cat, count in category_counts.items()
		}

		combined_distribution = {
			combo: {
				"count": count,
				"percentage": (count / total_count) * 100,
			}
			for combo, count in combined_counts.items()
		}

		# Target turns statistics
		target_turns_stats = {}
		if target_turns:
			target_turns_stats = {
				"mean": sum(target_turns) / len(target_turns),
				"min": min(target_turns),
				"max": max(target_turns),
				"distribution": dict(Counter(target_turns)),
			}

		# Compute "averages" as mentioned in PLAN.md (diversity metrics)
		specificity_diversity = len(specificity_counts)  # Number of unique specificities
		category_diversity = len(category_counts)  # Number of unique categories
		combined_diversity = len(combined_counts)  # Number of unique combinations

		return {
			"total_conversations": len(individual_results),
			"successful_evaluations": len(successful_results),
			"success_rate": len(successful_results) / len(individual_results),
			# Distribution data (for histograms)
			"specificity_distribution": specificity_distribution,
			"category_distribution": category_distribution,
			"combined_distribution": combined_distribution,
			# Diversity metrics ("averages" as mentioned in PLAN.md)
			"specificity_diversity": specificity_diversity,
			"category_diversity": category_diversity,
			"combined_diversity": combined_diversity,
			# Target turns analysis
			"target_turns_stats": target_turns_stats,
			# Raw data for further analysis
			"raw_specificities": specificities,
			"raw_categories": categories,
			"raw_combined": combined_specs_cats,
			# Summary statistics
			"most_common_specificity": specificity_counts.most_common(1)[0] if specificity_counts else None,
			"most_common_category": category_counts.most_common(1)[0] if category_counts else None,
			"most_common_combination": combined_counts.most_common(1)[0] if combined_counts else None,
		}


# Create the evaluator instance for easy import
conversation_goal_distribution_evaluator = ConversationGoalDistributionEvaluator()
