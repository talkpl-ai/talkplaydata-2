from typing import Any

from google import genai

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import call_with_timeout, extract_detailed_token_usage, robust_parse_yaml_response
from tp2dg.entities import ConversationGoal
from tp2dg.entities.reponse_code import ListenerProfileCode
from tp2dg.entities.token_usage import TokenUsage
from tp2dg.entities.track import Tracks
from tp2dg.prompts.profile_llm.query import profile_query


class ProfileLLM(BaseComponent):
	def __init__(self, client: genai.Client, model: str, api_delay: float, profile_information):
		super().__init__(api_delay)
		self.client = client
		self.model = model
		self.last_prompt = ""
		self.last_token_usage = TokenUsage()
		self.profile_information = profile_information

	def get_last_token_usage(self) -> TokenUsage:
		return self.last_token_usage

	def generate_from_tracks(
		self,
		listener_tracks: Tracks,
		uploaded_audio_files: dict[str, Any],
		uploaded_image_files: dict[str, Any],
	):
		track_contents = listener_tracks.prompt_str_with_artifacts(
			tracks_artifacts={"audio": uploaded_audio_files, "image": uploaded_image_files},
			include_track_id=False,
		)
		inference_prompt = profile_query.format(
			age_group=self.profile_information["age_group"],
			country=self.profile_information["country"],
			gender=self.profile_information["gender"],
			preferred_language=self.profile_information["preferred_language"],
		)
		contents = [inference_prompt, *track_contents]
		response = call_with_timeout(
			lambda: self.client.models.generate_content(model=self.model, contents=contents),
			timeout=120,
		)
		self.last_prompt = str(contents)
		self.last_token_usage = extract_detailed_token_usage(response)
		parsed = robust_parse_yaml_response(response.text, profile_query.response_expected_fields)
		from tp2dg.entities.listener_profile import ListenerProfile
		return ListenerProfile(
			age_group=self.profile_information["age_group"],
			country=self.profile_information["country"],
			gender=self.profile_information["gender"],
			preferred_musical_culture=parsed.get("preferred_musical_culture", "Unknown"),
			preferred_language=self.profile_information["preferred_language"],
			top_1_artist=parsed.get("top_1_artist", "Unknown"),
			top_1_genre=parsed.get("top_1_genre", "Unknown"),
			success=True,
			code=ListenerProfileCode.SUCCESS,
		) 