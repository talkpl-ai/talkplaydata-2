from typing import Any

from google import genai

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import call_with_timeout, extract_detailed_token_usage, robust_parse_yaml_response
from tp2dg.entities.reponse_code import ListenerTurnCode
from tp2dg.entities.token_usage import TokenUsage
from tp2dg.entities.track import Track
from tp2dg.entities.turns import ListenerTurn
from tp2dg.prompts.listener_llm.query import listener_first_turn, reaction_turn2, reaction_turn_n
from tp2dg.prompts.listener_llm.system import listener_system, listener_turn_0


class ListenerLLM(BaseComponent):
	def __init__(self, client: genai.Client, model: str, api_delay: float = 0.0):
		super().__init__(api_delay)
		self.client = client
		self.model = model
		self.last_prompt = ""
		self.last_token_usage = TokenUsage()
		self.chat_session = None

	def set_chat_session(self, chat_session):
		self.chat_session = chat_session

	def initialize_session(self, listener_profile, conversation_goal, previously_liked_tracks, uploaded_audio_files, uploaded_image_files):
		listener_system_instruction = listener_system.format(
			listener_profile=listener_profile.prompt_str(), conversation_goal=conversation_goal.prompt_str()
		)
		self.chat_session = self.client.chats.create(model=self.model, config={"system_instruction": listener_system_instruction})
		liked_tracks_contents = previously_liked_tracks.prompt_str_with_artifacts(
			tracks_artifacts={"audio": uploaded_audio_files, "image": uploaded_image_files},
			include_track_id=True,
			title="## Your Previously Liked Tracks\n\n",
		)
		self.chat_session.send_message([*liked_tracks_contents, listener_turn_0.format()])

	def get_initial_request(self, initial_query_examples: list[str], listener_goal: str, preferred_language: str) -> ListenerTurn:
		if not self.chat_session:
			raise Exception("Listener chat session not initialized")
		initial_query_examples_text = "\n".join([f'- "{ex}"' for ex in initial_query_examples])
		prompt = listener_first_turn.format(
			initial_query_examples=initial_query_examples_text,
			listener_goal=listener_goal,
			preferred_language=preferred_language,
		)
		response = call_with_timeout(lambda: self.chat_session.send_message(prompt), timeout=120)
		self.last_prompt = prompt
		self.last_token_usage = extract_detailed_token_usage(response)
		parsed = robust_parse_yaml_response(response.text, listener_first_turn.response_expected_fields)
		return ListenerTurn(
			turn_number=1,
			prompt=prompt,
			thought=parsed.get("thought", ""),
			goal_progress_assessment=None,
			message=parsed.get("message", ""),
			success=True,
			code=ListenerTurnCode.SUCCESS,
		)

	def get_reaction_with_thought(
		self,
		turn_num: int,
		recommended_track: Track,
		uploaded_audio_files: dict[str, Any],
		uploaded_image_files: dict[str, Any],
		recsys_message: str,
		preferred_language: str,
	) -> ListenerTurn:
		if not self.chat_session:
			raise Exception("Listener chat session not initialized")
		track_audio_file = uploaded_audio_files.get(recommended_track.track_id, None)
		track_image_file = uploaded_image_files.get(recommended_track.track_id, None)
		if turn_num == 2:
			prompt_text = reaction_turn2.format(
				turn_num=turn_num,
				title=recommended_track.title,
				artist=recommended_track.artist,
				album=recommended_track.album,
				recsys_message=recsys_message,
				preferred_language=preferred_language,
			)
		else:
			prompt_text = reaction_turn_n.format(
				turn_num=turn_num,
				title=recommended_track.title,
				artist=recommended_track.artist,
				album=recommended_track.album,
				recsys_message=recsys_message,
				preferred_language=preferred_language,
			)
		contents = []
		if track_audio_file:
			contents.append(track_audio_file)
		if track_image_file:
			contents.append(track_image_file)
		contents.append(prompt_text)
		response = call_with_timeout(lambda: self.chat_session.send_message(contents), timeout=120)
		self.last_prompt = prompt_text
		self.last_token_usage = extract_detailed_token_usage(response)
		expected = reaction_turn_n.response_expected_fields
		parsed = robust_parse_yaml_response(response.text, expected)
		gpa = str(parsed.get("goal_progress_assessment", "")).strip().upper()
		return ListenerTurn(
			turn_number=turn_num,
			prompt=prompt_text,
			thought=parsed.get("thought", ""),
			goal_progress_assessment=gpa,
			message=parsed.get("message", ""),
			success=True,
			code=ListenerTurnCode.SUCCESS,
		) 