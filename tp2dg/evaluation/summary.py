import argparse
import json
import os
from glob import glob


def parse_args():
    p = argparse.ArgumentParser(description="Summarize generated conversations")
    p.add_argument("--input", type=str, required=True, help="Root folder with generated_conversations")
    return p.parse_args()


def load_chats(root: str):
    pattern = os.path.join(root, "**", "chat.json")
    for path in glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                yield path, json.load(f)
        except Exception:
            continue


def main():
    args = parse_args()
    total = 0
    total_turns = 0

    for path, chat in load_chats(args.input):
        total += 1
        total_turns += len(chat)

    if total == 0:
        print("No conversations found.")
        return

    avg_turns = total_turns / total
    print(f"Conversations: {total}")
    print(f"Average turns: {avg_turns:.2f}")


if __name__ == "__main__":
    main() 