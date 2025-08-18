from typing import Any

from google import genai

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import call_with_timeout, robust_parse_yaml_response
from tp2dg.entities.conversation_goal import ConversationGoal
from tp2dg.entities.track import Tracks
from tp2dg.prompts.conversation_goal.query import conversation_goal_query_pt1, conversation_goal_query_pt2

GOALS_TO_SAMPLE = 3


class ConversationGoalLLM(BaseComponent):
	def __init__(self, client: genai.Client, model: str, api_delay: float = 0.0):
		super().__init__(api_delay)
		self.client = client
		self.model = model
		self.last_prompt = ""

	def generate_from_recommendation_pool(
		self,
		recommendation_pool: Tracks,
		uploaded_audio_files: dict[str, Any],
		uploaded_image_files: dict[str, Any],
		seed: int,
	) -> ConversationGoal:
		prompt_pt1 = conversation_goal_query_pt1.format(number_of_conversation_goals=GOALS_TO_SAMPLE)
		track_contents = recommendation_pool.prompt_str_with_artifacts(
			include_track_id=False, tracks_artifacts={"audio": uploaded_audio_files, "image": uploaded_image_files}
		)
		conversation_goals = ConversationGoal.sample_conversation_goals(seed, GOALS_TO_SAMPLE)
		prompt_pt2 = conversation_goal_query_pt2.format(conversation_goal_templates=conversation_goals.prompt_str())
		contents = [prompt_pt1, *track_contents, prompt_pt2]
		response = call_with_timeout(lambda: self.client.models.generate_content(model=self.model, contents=contents), timeout=180)
		parsed = robust_parse_yaml_response(response.text, conversation_goal_query_pt2.response_expected_fields)
		try:
			return ConversationGoal.from_codes(
				ConversationGoal.CategoryCode(parsed["category_code"]), ConversationGoal.SpecificityCode(parsed["specificity_code"])
			)
		except Exception:
			return ConversationGoal.unknown_conversation_goal() 