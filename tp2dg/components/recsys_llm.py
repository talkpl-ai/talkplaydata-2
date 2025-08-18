from google import genai

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import call_with_timeout, extract_detailed_token_usage, robust_parse_yaml_response
from tp2dg.entities import ConversationTurns, RecsysTurn
from tp2dg.entities.reponse_code import RecsysTurnCode
from tp2dg.entities.token_usage import TokenUsage
from tp2dg.entities.track import Tracks
from tp2dg.prompts.recsys_llm.query import recsys_following_turns
from tp2dg.prompts.recsys_llm.system import recsys_system, recsys_turn_0_pt1, recsys_turn_0_pt2


class RecsysLLM(BaseComponent):
	def __init__(self, client: genai.Client, model: str, api_delay: float = 0.0):
		super().__init__(api_delay)
		self.client = client
		self.model = model
		self.last_prompt = ""
		self.last_token_usage = TokenUsage()
		self.chat_session = None

	def set_chat_session(self, chat_session):
		self.chat_session = chat_session

	def initialize_session(self, listener_profile, recommendation_pool: Tracks, uploaded_audio_files, uploaded_image_files):
		recsys_system_instruction = recsys_system.format()
		self.chat_session = self.client.chats.create(model=self.model, config={"system_instruction": recsys_system_instruction})
		artifacts = {"audio": uploaded_audio_files, "image": uploaded_image_files}
		turn0_pt1_prompt = recsys_turn_0_pt1.format(listener_profile=listener_profile.prompt_str())
		available_tracks_contents = recommendation_pool.prompt_str_with_artifacts(
			tracks_artifacts=artifacts,
			include_track_id=True,
			tracks_title="## RECOMMENDATION POOL\n\n",
		)
		turn0_pt2_prompt = recsys_turn_0_pt2.format()
		turn0_contents = [turn0_pt1_prompt, *available_tracks_contents, turn0_pt2_prompt]
		self.chat_session.send_message(turn0_contents)

	def get_recommendation_with_thought(
		self,
		*,
		turn_num: int,
		conversation_turns: ConversationTurns,
		available_tracks: Tracks,
		listener_message: str,
		preferred_language: str,
	):
		if not self.chat_session:
			raise Exception("RecSys chat session not initialized")
		used_track_ids = conversation_turns.used_track_ids() if hasattr(conversation_turns, "used_track_ids") else []
		prompt = recsys_following_turns.format(
			turn_num=turn_num,
			used_track_ids=used_track_ids,
			listener_message=listener_message,
			preferred_language=preferred_language,
		)
		response = call_with_timeout(lambda: self.chat_session.send_message(prompt), timeout=120)
		self.last_prompt = prompt
		self.last_token_usage = extract_detailed_token_usage(response)
		parsed = robust_parse_yaml_response(response.text, recsys_following_turns.response_expected_fields)
		thought = parsed.get("thought", "Unknown thought")
		track_id = parsed.get("track_id", "")
		message = parsed.get("message", "Unknown message")
		recommended_track = None
		for track in available_tracks:
			if track.track_id == track_id:
				recommended_track = track
				break
		if recommended_track is None:
			return RecsysTurn(
				turn_number=turn_num,
				prompt=prompt,
				thought="No match",
				track=None,
				message="No track found",
				success=False,
				code=RecsysTurnCode.NO_TRACK_FOUND,
			)
		return RecsysTurn(
			turn_number=turn_num,
			prompt=prompt,
			thought=thought,
			track=recommended_track,
			message=message,
			success=True,
			code=RecsysTurnCode.SUCCESS,
		) 