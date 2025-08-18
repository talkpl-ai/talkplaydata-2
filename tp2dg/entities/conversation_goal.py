import os
import random
from collections import UserList
from dataclasses import dataclass
from enum import Enum

import yaml

CONVERSATION_GOALS = []
CONVERSATION_GOALS_YAML_PATH = os.path.join(os.path.dirname(__file__), "conversation_goals.yaml")

with open(CONVERSATION_GOALS_YAML_PATH) as f:
	CONVERSATION_GOALS = yaml.safe_load(f)


class ConversationGoalCategoryCode(Enum):
	A = "A"
	B = "B"
	C = "C"
	D = "D"
	E = "E"
	F = "F"
	G = "G"
	H = "H"
	I = "I"
	J = "J"
	K = "K"
	UNKNOWN = "UNKNOWN"

	@classmethod
	def get_selectable_codes(cls) -> list["ConversationGoalCategoryCode"]:
		return [c for c in cls if c.name != "UNKNOWN"]


class ConversationGoalSpecificityCode(Enum):
	LL = "LL"
	LH = "LH"
	HH = "HH"
	HL = "HL"
	UNKNOWN = "UNKNOWN"

	@classmethod
	def get_selectable_codes(cls) -> list["ConversationGoalSpecificityCode"]:
		return [c for c in cls if c.name != "UNKNOWN"]


@dataclass
class ConversationGoal:
	category_code: ConversationGoalCategoryCode
	category_description: str
	specificity_code: ConversationGoalSpecificityCode
	specificity_description: str
	listener_goal: str
	listener_expertise: str
	initial_query_examples: list[str]
	iteration_query_examples: list[str]
	achieved_query_examples: list[str]
	target_turn_count: int = 8
	success: bool = True

	@classmethod
	def list_all_category_codes(cls) -> list[ConversationGoalCategoryCode]:
		return list(ConversationGoalCategoryCode)

	@classmethod
	def list_all_specificity_codes(cls) -> list[ConversationGoalSpecificityCode]:
		return list(ConversationGoalSpecificityCode)

	@classmethod
	def find_conversation_goal(cls, category: str, specificity: str) -> dict:
		for goal in CONVERSATION_GOALS:
			if goal["category"]["code"] == category and goal["specificity"]["code"] == specificity:
				return goal
		raise ValueError(f"No conversation goal found for category {category} and specificity {specificity}")

	@classmethod
	def sample_conversation_goals(cls, seed: int, num_goals: int) -> "ConversationGoals":
		if num_goals <= 0:
			raise ValueError(f"Cannot sample {num_goals} goals.")
		random.seed(seed)
		selectable = [
			(c.value, s.value)
			for c in ConversationGoalCategoryCode.get_selectable_codes()
			for s in ConversationGoalSpecificityCode.get_selectable_codes()
		]
		random.shuffle(selectable)
		selected = []
		for cat, spe in selectable:
			g = cls.find_conversation_goal(cat, spe)
			selected.append(
				ConversationGoal.from_codes(
					ConversationGoalCategoryCode(cat), ConversationGoalSpecificityCode(spe)
				),
			)
			if len(selected) >= num_goals:
				break
		return ConversationGoals(selected)

	@classmethod
	def unknown_conversation_goal(cls) -> "ConversationGoal":
		return cls(
			category_code=ConversationGoalCategoryCode.UNKNOWN,
			category_description="Unknown",
			specificity_code=ConversationGoalSpecificityCode.UNKNOWN,
			specificity_description="Unknown",
			listener_goal="Unknown",
			listener_expertise="Unknown",
			initial_query_examples=["Unknown"],
			iteration_query_examples=["Unknown"],
			achieved_query_examples=["Unknown"],
			target_turn_count=8,
			success=False,
		)

	@classmethod
	def from_codes(
		cls,
		category_code: ConversationGoalCategoryCode,
		specificity_code: ConversationGoalSpecificityCode,
	) -> "ConversationGoal":
		if (
			category_code == ConversationGoalCategoryCode.UNKNOWN
			or specificity_code == ConversationGoalSpecificityCode.UNKNOWN
		):
			return cls.unknown_conversation_goal()
		goal = cls.find_conversation_goal(category_code.value, specificity_code.value)
		return cls(
			category_code=category_code,
			category_description=goal["category"]["description"],
			specificity_code=specificity_code,
			specificity_description=goal["specificity"]["description"],
			listener_goal=goal["listener_goal"],
			listener_expertise=goal["listener_expertise"],
			initial_query_examples=[d["example"] for d in goal["queries"]["initial"]],
			iteration_query_examples=[d["example"] for d in goal["queries"]["iteration"]],
			achieved_query_examples=[d["example"] for d in goal["queries"]["achieved"]],
			target_turn_count=8,
		)

	def prompt_str(self, title: str = "## Conversation Goal\n\n") -> str:
		return f"{title}- Category: {self.category_code.value}\n- Category Description: {self.category_description}\n- Specificity: {self.specificity_code.value}\n- Specificity Description: {self.specificity_description}\n- Listener goal: {self.listener_goal}\n- Listener expertise: {self.listener_expertise}\n- Target turn count: {self.target_turn_count}\n- Initial query example: {', '.join(self.initial_query_examples)}\n- Iteration query example: {', '.join(self.iteration_query_examples)}\n- Achieved query example: {', '.join(self.achieved_query_examples)}"


class ConversationGoals(UserList):
	def __init__(self, data: list[ConversationGoal]):
		super().__init__(data)

	def prompt_str(self, title: str = "# Allowed Conversation Goals\n\n") -> str:
		goals_str = [goal.prompt_str() for goal in self.data]
		goals_str = "\n\n".join(goals_str)
		return f"{title}{goals_str}" 