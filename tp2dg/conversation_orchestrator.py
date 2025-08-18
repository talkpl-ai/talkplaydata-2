import os
import random
from typing import Any, Dict

from google import genai

from tp2dg.components.chat_session_manager import ChatSessionManager
from tp2dg.components.conversation_goal_llm import ConversationGoalLLM
from tp2dg.components.listener_llm import ListenerLLM
from tp2dg.components.processors import AudioProcessor, ImageProcessor
from tp2dg.components.profile_llm import ProfileLLM
from tp2dg.components.recsys_llm import RecsysLLM
from tp2dg.entities.turns import ConversationTurn, ConversationTurns
from tp2dg.entities.track import Tracks


class ConversationOrchestrator:
	def __init__(self, model: str, snippet_duration: float = 0.0, audio_base_path: str = "", image_base_path: str = "", api_delay: float = 0.0, profile_information: Dict | None = None, seed: int = 42):
		self.model = model
		self.snippet_duration = snippet_duration
		self.api_delay = api_delay
		self.profile_information = profile_information or {"age_group": "20s", "country": "US", "gender": "male", "preferred_language": "English"}
		self.shared_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
		self.audio_processor = AudioProcessor(client=self.shared_client, audio_base_path=audio_base_path, snippet_duration=self.snippet_duration)
		self.image_processor = ImageProcessor(client=self.shared_client, image_base_path=image_base_path)
		self.chat_manager = ChatSessionManager(client=self.shared_client, model=model)
		self.profile_llm = ProfileLLM(client=self.shared_client, model=self.model, api_delay=self.api_delay, profile_information=self.profile_information)
		self.conversation_goal_llm = ConversationGoalLLM(client=self.shared_client, model=self.model, api_delay=self.api_delay)
		self.recsys_llm = RecsysLLM(client=self.shared_client, model=self.model, api_delay=self.api_delay)
		self.listener_llm = ListenerLLM(client=self.shared_client, model=self.model, api_delay=self.api_delay)
		random.seed(seed)

	def generate(self, user: Dict, liked: Tracks, pool: Tracks, num_turns: int = 4) -> Dict:
		# Upload artifacts
		uploaded_audio_files = self.audio_processor.batch_upload_tracks(liked + pool)
		uploaded_image_files = self.image_processor.batch_upload_tracks(liked + pool)
		# Profile and goal
		listener_profile = self.profile_llm.generate_from_tracks(liked, uploaded_audio_files, uploaded_image_files)
		conversation_goal = self.conversation_goal_llm.generate_from_recommendation_pool(pool, uploaded_audio_files, uploaded_image_files, seed=42)
		# Initialize chats
		self.chat_manager.initialize_recsys_session(listener_profile, pool, uploaded_audio_files, uploaded_image_files)
		self.chat_manager.initialize_listener_session(listener_profile, conversation_goal, liked, uploaded_audio_files, uploaded_image_files)
		self.recsys_llm.set_chat_session(self.chat_manager.recsys_chat)
		self.listener_llm.set_chat_session(self.chat_manager.listener_chat)
		# Conversation loop
		conversation = ConversationTurns()
		available = pool[:]
		listener_turn = self.listener_llm.get_initial_request(conversation_goal.initial_query_examples, conversation_goal.listener_goal, listener_profile.preferred_language)
		for turn_num in range(1, num_turns + 1):
			if not available:
				break
			recsys_turn = self.recsys_llm.get_recommendation_with_thought(
				turn_num=turn_num,
				conversation_turns=conversation,
				available_tracks=available,
				listener_message=listener_turn.message,
				preferred_language=listener_profile.preferred_language,
			)
			conversation.append(ConversationTurn(turn_number=turn_num, listener_turn=listener_turn, recsys_turn=recsys_turn))
			# remove used
			available = [t for t in available if not (recsys_turn.track and t.track_id == recsys_turn.track.track_id)]
			if turn_num < num_turns and recsys_turn.track:
				listener_turn = self.listener_llm.get_reaction_with_thought(
					turn_num + 1,
					recsys_turn.track,
					uploaded_audio_files,
					uploaded_image_files,
					recsys_turn.message,
					listener_profile.preferred_language,
				)
		return {
			"profiling": {"user": user, "summary": listener_profile.prompt_str()},
			"conversation_goal": {"goal": conversation_goal.listener_goal, "examples": conversation_goal.initial_query_examples[:2]},
			"chat": [
				{
					"turn": tr.turn_number,
					"listener": {
						"thought": tr.listener_turn.thought,
						"goal_progress_assessment": tr.listener_turn.goal_progress_assessment,
						"message": tr.listener_turn.message,
					},
					"recsys": {
						"thought": tr.recsys_turn.thought,
						"message": tr.recsys_turn.message,
						"track": tr.recsys_turn.track.__dict__ if tr.recsys_turn.track else None,
					},
				}
				for tr in conversation
			],
		}

	def save_outputs(self, outputs: Dict, out_dir: str) -> None:
		os.makedirs(out_dir, exist_ok=True)
		import json

		with open(os.path.join(out_dir, "profiling.json"), "w", encoding="utf-8") as f:
			json.dump(outputs["profiling"], f, indent=2, ensure_ascii=False)
		with open(os.path.join(out_dir, "conversation_goal.json"), "w", encoding="utf-8") as f:
			json.dump(outputs["conversation_goal"], f, indent=2, ensure_ascii=False)
		with open(os.path.join(out_dir, "chat.json"), "w", encoding="utf-8") as f:
			json.dump(outputs["chat"], f, indent=2, ensure_ascii=False) 