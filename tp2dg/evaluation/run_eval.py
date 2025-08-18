import argparse
import json
import os
from collections import Counter, defaultdict
from glob import glob

from google import genai

from tp2dg.components.utils import robust_parse_yaml_response
from tp2dg.evaluation.prompts.conversation_goal.plausibility import (
	conversation_goal_plausibility_evaluator,
)
from tp2dg.evaluation.prompts.conversation_element.thought import thought_evaluator
from tp2dg.evaluation.prompts.conversation_element.message import message_evaluator
from tp2dg.evaluation.prompts.conversation_element.goal_progress_assessment import (
	goal_progress_assessment_evaluator,
)
from tp2dg.evaluation.prompts.conversation_element.track_id import track_id_evaluator


def parse_args():
	p = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation over generated conversations")
	p.add_argument("--input", required=True, help="Root folder with generated conversations")
	p.add_argument("--model", default="gemini-2.5-flash")
	return p.parse_args()


def load_conversations(root: str):
	for chat_path in glob(os.path.join(root, "**", "chat.json"), recursive=True):
		base = os.path.dirname(chat_path)
		with open(chat_path, "r", encoding="utf-8") as f:
			chat_list = json.load(f)
		goal = {}
		profile = {}
		try:
			with open(os.path.join(base, "conversation_goal.json"), "r", encoding="utf-8") as f:
				goal = json.load(f)
			with open(os.path.join(base, "profiling.json"), "r", encoding="utf-8") as f:
				profile = json.load(f)
		except Exception:
			pass
		yield {"chat_list": chat_list, "goal": goal, "profile": profile, "base": base}


def to_score(text: str, key: str) -> int:
	parsed = robust_parse_yaml_response(text, [key])
	val = parsed.get(key)
	try:
		return int(val)
	except Exception:
		# fallback: first digit
		for ch in str(text):
			if ch in "1234":
				return int(ch)
	return 0


def adapt_to_eval_structure(chat_list: list, goal: dict, profile: dict) -> dict:
	conversation_turns = []
	for item in chat_list:
		conversation_turns.append(
			{
				"turn_number": item.get("turn", 0),
				"listener_turn": {
					"thought": item.get("listener", {}).get("thought", ""),
					"message": item.get("listener", {}).get("message", ""),
					"goal_progress_assessment": item.get("listener", {}).get("goal_progress_assessment", ""),
				},
				"recsys_turn": {
					"thought": item.get("recsys", {}).get("thought", ""),
					"message": item.get("recsys", {}).get("message", ""),
					"track": item.get("recsys", {}).get("track", None),
				},
			}
		)
	# Minimal session context placeholders (prompts are verbatim; content may be empty)
	session_context = {
		"recommendation_pool_tracks": [],
		"listener_tracks": [],
	}
	return {
		"conversation_turns": conversation_turns,
		"conversation_goal": goal or {},
		"listener_profile": profile.get("summary", {}),
		"session_context": session_context,
	}


def main():
	args = parse_args()
	client = genai.Client()
	model = args.model

	metrics = defaultdict(Counter)

	for convo in load_conversations(args.input):
		chat_list = convo["chat_list"]
		goal = convo["goal"]
		profile = convo["profile"]

		chat_json = adapt_to_eval_structure(chat_list, goal, profile)

		# Goal plausibility (if goal available)
		if goal:
			pd = conversation_goal_plausibility_evaluator.prepare_prompt_data(chat_json, {}, {})
			contents = conversation_goal_plausibility_evaluator.prompt_template.format(**pd)
			resp = client.models.generate_content(model=model, contents=contents)
			s = to_score(resp.text, "plausibility_score")
			metrics["goal_plausibility"][s] += 1

		# Profile appropriateness (if profile available) - reuse message evaluator's listener_quality_score as proxy is not right.
		# Skipping explicit profile judge since dataset lacks per-track profiling context here.

		# Listener goal progress label accuracy per turn
		for turn in chat_json["conversation_turns"]:
			pd = goal_progress_assessment_evaluator.prepare_prompt_data(chat_json, {}, {})
			contents = goal_progress_assessment_evaluator.prompt_template.format(**pd)
			resp = client.models.generate_content(model=model, contents=contents)
			s = to_score(resp.text, "accuracy_score")
			metrics["listener_progress_label"][s] += 1
			break  # evaluate once per conversation to reduce cost

		# Thought quality (listener/recsys)
		pd = thought_evaluator.prepare_prompt_data(chat_json)
		contents = thought_evaluator.prompt_template.format(**pd)
		resp = client.models.generate_content(model=model, contents=contents)
		metrics["listener_thought_quality"][to_score(resp.text, "listener_coherence_score")] += 1
		metrics["recsys_thought_quality"][to_score(resp.text, "recsys_coherence_score")] += 1

		# Message quality and alignment (listener/recsys)
		pd = message_evaluator.prepare_prompt_data(chat_json, {}, {})
		contents = message_evaluator.prompt_template.format(**pd)
		resp = client.models.generate_content(model=model, contents=contents)
		metrics["listener_message_quality"][to_score(resp.text, "listener_quality_score")] += 1
		metrics["recsys_message_quality"][to_score(resp.text, "recsys_quality_score")] += 1
		metrics["listener_message_helpfulness"][to_score(resp.text, "listener_helpfulness_score")] += 1
		metrics["recsys_message_alignment"][to_score(resp.text, "recsys_accuracy_score")] += 1

		# Track_id recommendation quality (evaluate first turn only to reduce cost)
		pd = track_id_evaluator.prepare_prompt_data(chat_json, {}, {})
		contents = track_id_evaluator.prompt_template.format(**pd)
		resp = client.models.generate_content(model=model, contents=contents)
		metrics["recsys_track_quality"][to_score(resp.text, "recommendation_score")] += 1

	# Print distributions
	for k, counter in metrics.items():
		total = sum(counter.values())
		avg = (sum(score * count for score, count in counter.items()) / total) if total else 0
		dist = ", ".join(f"{s}:{counter.get(s,0)}" for s in [1, 2, 3, 4])
		print(f"{k}: avg={avg:.2f}/4, dist=({dist}), n={total}")


if __name__ == "__main__":
	main() 