from dataclasses import dataclass
from typing import Optional

from tp2dg.entities.reponse_code import ListenerTurnCode, RecsysTurnCode
from tp2dg.entities.track import Track


@dataclass
class ListenerTurn:
	turn_number: int
	prompt: str
	thought: str
	goal_progress_assessment: Optional[str]
	message: str
	success: bool = True
	code: str = ListenerTurnCode.SUCCESS

	VALID_GOAL_PROGRESS_ASSESSMENT = ["MOVES_TOWARD_GOAL", "DOES_NOT_MOVE_TOWARD_GOAL"]


@dataclass
class RecsysTurn:
	turn_number: int
	prompt: str
	thought: str
	track: Optional[Track]
	message: str
	success: bool = True
	code: str = RecsysTurnCode.SUCCESS


@dataclass
class ConversationTurn:
	turn_number: int
	listener_turn: ListenerTurn
	recsys_turn: RecsysTurn


class ConversationTurns(list):
	def used_track_ids(self) -> list[str]:
		return [turn.recsys_turn.track.track_id for turn in self if turn.recsys_turn.track is not None]

	def to_list_of_dicts(self) -> list[dict]:
		return [
			{
				"turn_number": tr.turn_number,
				"listener_turn": {
					"prompt": tr.listener_turn.prompt,
					"thought": tr.listener_turn.thought,
					"goal_progress_assessment": tr.listener_turn.goal_progress_assessment,
					"message": tr.listener_turn.message,
				},
				"recsys_turn": {
					"prompt": tr.recsys_turn.prompt,
					"thought": tr.recsys_turn.thought,
					"track": tr.recsys_turn.track.__dict__ if tr.recsys_turn.track else None,
					"message": tr.recsys_turn.message,
				},
			}
			for tr in self
		] 