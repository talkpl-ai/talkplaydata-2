from google import genai
from google.genai import types

from tp2dg.components.base import BaseComponent
from tp2dg.components.utils import extract_detailed_token_usage
from tp2dg.entities.token_usage import TokenUsage
from tp2dg.entities.track import Tracks
from tp2dg.prompts.listener_llm.system import listener_system, listener_turn_0
from tp2dg.prompts.recsys_llm.system import recsys_system, recsys_turn_0_pt1, recsys_turn_0_pt2


class ChatSessionManager(BaseComponent):
	def __init__(self, client: genai.Client, model: str):
		super().__init__()
		self.client = client
		self.model = model
		self.recsys_interactions = []
		self.listener_interactions = []
		self.profiling_interactions = []
		self.conversation_goal_interactions = []
		self.recsys_chat = None
		self.listener_chat = None

	def initialize_recsys_session(self, listener_profile, recommendation_pool: Tracks, uploaded_audio_files=None, uploaded_image_files=None):
		if uploaded_audio_files is None:
			uploaded_audio_files = {}
		if uploaded_image_files is None:
			uploaded_image_files = {}
		recsys_system_instruction = recsys_system.format()
		self.recsys_chat = self.client.chats.create(
			model=self.model,
			config=types.GenerateContentConfig(system_instruction=recsys_system_instruction),
		)
		self.recsys_interactions.append(
			{"turn": "None", "type": "system_initialization", "prompt": recsys_system_instruction, "response": "System initialized", "token_usage": TokenUsage().to_dict()},
		)
		artifacts = {"audio": uploaded_audio_files, "image": uploaded_image_files}
		turn0_pt1_prompt = recsys_turn_0_pt1.format(listener_profile=listener_profile.prompt_str())
		available_tracks_contents = recommendation_pool.prompt_str_with_artifacts(
			tracks_artifacts=artifacts, include_track_id=True, tracks_title="## RECOMMENDATION POOL\n\n"
		)
		turn0_pt2_prompt = recsys_turn_0_pt2.format()
		turn0_contents = [turn0_pt1_prompt, *available_tracks_contents, turn0_pt2_prompt]
		recsys_response = self.recsys_chat.send_message(turn0_contents)
		recsys_token_usage = extract_detailed_token_usage(recsys_response)
		self.recsys_interactions.append(
			{"turn": 0, "type": "system_initialization_with_audio", "prompt": str(turn0_contents), "response": recsys_response.text if recsys_response else "No response", "token_usage": recsys_token_usage.to_dict(), "total_tracks_analyzed": len(recommendation_pool)},
		)

	def initialize_listener_session(self, listener_profile, conversation_goal, previously_liked_tracks: Tracks, uploaded_audio_files=None, uploaded_image_files=None):
		if uploaded_audio_files is None:
			uploaded_audio_files = {}
		if uploaded_image_files is None:
			uploaded_image_files = {}
		listener_system_instruction = listener_system.format(
			listener_profile=listener_profile.prompt_str(), conversation_goal=conversation_goal.prompt_str()
		)
		self.listener_chat = self.client.chats.create(
			model=self.model,
			config=types.GenerateContentConfig(system_instruction=listener_system_instruction),
		)
		self.listener_interactions.append(
			{"turn": "None", "type": "system_initialization", "prompt": listener_system_instruction, "response": "System initialized", "token_usage": TokenUsage().to_dict()},
		)
		if len(previously_liked_tracks) > 0:
			artifacts = {"audio": uploaded_audio_files, "image": uploaded_image_files}
			liked_tracks_contents = previously_liked_tracks.prompt_str_with_artifacts(
				tracks_artifacts=artifacts, include_track_id=True, tracks_title="## Your Previously Liked Tracks\n\n"
			)
			contents = [*liked_tracks_contents, listener_turn_0.format()]
			listener_response = self.listener_chat.send_message(contents)
			listener_token_usage = extract_detailed_token_usage(listener_response)
			self.listener_interactions.append(
				{"turn": 0, "type": "previously_liked_tracks", "prompt": str(liked_tracks_contents), "response": listener_response.text if listener_response else "No response", "token_usage": listener_token_usage.to_dict()},
			)

	def get_all_interactions(self) -> dict[str, list]:
		return {
			"profiling": self.profiling_interactions,
			"conversation_goal": self.conversation_goal_interactions,
			"recsys": self.recsys_interactions,
			"listener": self.listener_interactions,
		} 