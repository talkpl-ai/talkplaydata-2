import argparse
import os

from .data.loader import get_first_session
from .conversation_orchestrator import ConversationOrchestrator


def parse_args():
    p = argparse.ArgumentParser(description="Generate a tiny demo conversation with dummy data (Gemini-only).")
    p.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    p.add_argument("--output-dir", type=str, default="generated_conversations", help="Output directory")
    p.add_argument("--turns", type=int, default=4, help="Number of turns")
    return p.parse_args()


def main():
    args = parse_args()

    sess = get_first_session()

    orch = ConversationOrchestrator(model=args.model, seed=42)
    outputs = orch.generate(user=sess.user, liked=sess.liked_tracks, pool=sess.pool_tracks, num_turns=args.turns)

    out_dir = os.path.join(args.output_dir, args.model, "dummy", sess.user["user_id"], sess.session_id)
    orch.save_outputs(outputs, out_dir)
    print(f"Saved conversation to: {out_dir}")


if __name__ == "__main__":
    main() 