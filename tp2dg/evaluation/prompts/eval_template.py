"""
Abstract evaluation template for TalkPlayData 2 evaluation system.
Defines the structure for LLM judge evaluations and result aggregation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseEvaluationTemplate(ABC):
	"""
	Base class for all evaluation templates.

	Provides common functionality for LLM-based evaluations and defines
	the interface that all evaluators should implement.
	"""

	def __init__(self):
		self.evaluation_name = "base_evaluation"
		self.prompt_template = None
		self.logger = logging.getLogger(self.__class__.__name__)

	@abstractmethod
	def prepare_prompt_data(self, chat_json: dict[str, Any]) -> dict[str, Any]:
		"""
		Prepare the data needed for the evaluation prompt.

		Args:
			chat_json: Raw conversation data from chat.json

		Returns:
			Dictionary containing formatted data for the prompt template
		"""

	@abstractmethod
	async def evaluate_single(
		self,
		chat_json: dict[str, Any],
		llm_call_func,
		client,
		uploaded_audio_files: Optional[dict[str, Any]] = None,
		uploaded_image_files: Optional[dict[str, Any]] = None,
	) -> dict[str, Any]:
		"""
		Evaluate a single conversation.

		Args:
			chat_json: Raw conversation data from chat.json
			llm_call_func: Async function to call LLM with prompt
			client: Gemini client
			uploaded_audio_files: Dictionary of uploaded audio files
			uploaded_image_files: Dictionary of uploaded image files

		Returns:
			Dictionary containing evaluation result
		"""

	@abstractmethod
	def aggregate_results(self, individual_results: list[dict[str, Any]]) -> dict[str, Any]:
		"""
		Aggregate individual evaluation results into summary statistics.

		Args:
			individual_results: List of individual evaluation results

		Returns:
			Dictionary containing aggregated metrics and statistics
		"""
 