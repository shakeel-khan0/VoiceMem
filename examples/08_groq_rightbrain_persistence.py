"""Groq + local E5 Right Brain persistence smoke test.

Run from the repository root in two separate processes:

    python examples/08_groq_rightbrain_persistence.py ingest
    python examples/08_groq_rightbrain_persistence.py search
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Run directly from a source clone without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voicemem import VoiceMem


CONVERSATION = (
    "My name is Ali Khan. "
    "I run a dental clinic. "
    "I prefer evening meetings because mornings are usually busy. "
    "I get frustrated when we miss customer calls after business hours."
)
RIGHT_QUERY = "When should meetings with Ali be scheduled, and what frustrates him?"
LEFT_QUERY = "What business does Ali Khan run?"
MEMORY_ROOT = Path(__file__).resolve().parent / "groq_rightbrain_memory"


def make_voicemem() -> VoiceMem:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required")
    return VoiceMem.from_config({
        # Text mode keeps Right Brain enabled without loading audio components.
        "mode": "text",
        "memory_root": str(MEMORY_ROOT),
        "llm": {"provider": "groq"},
        "embedding": {"provider": "local"},
        "slots": {"provider": "local"},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("ingest", "search"))
    args = parser.parse_args()

    vm = make_voicemem()
    if args.phase == "ingest":
        result = vm.ingest(CONVERSATION)
        if not result.get("memory_ids"):
            raise SystemExit("FAIL: Left Brain extracted no memories")

        # A Right Brain heartnote is independent of Left Brain memory IDs, so
        # verify the public Right Brain store rather than assuming one was made.
        right_memories = vm.right_brain._rb_repo()._store.get_all("voice_user")
        if not right_memories:
            raise SystemExit("FAIL: Right Brain stored no memories")
        print(
            f"PASS: persisted {len(result['memory_ids'])} Left Brain facts and "
            f"{len(right_memories)} Right Brain memories"
        )
        return

    right_result = vm.search(RIGHT_QUERY, top_k=8)
    right_text = "\n".join(right_result.result_rightbrain)
    print("[Right Brain]")
    print(right_text)
    normalized = right_text.lower()
    if "evening" not in normalized or "frustrat" not in normalized or "call" not in normalized:
        raise SystemExit("FAIL: Right Brain did not retrieve meeting preference and missed-call frustration")

    left_result = vm.search(LEFT_QUERY, top_k=5)
    left_text = "\n".join(left_result.result_leftbrain)
    print("[Left Brain]")
    print(left_text)
    if "dental clinic" not in left_text.lower():
        raise SystemExit("FAIL: unchanged Left Brain did not retrieve Ali's business")

    print("PASS: retrieved persisted Right Brain context and unchanged Left Brain facts")


if __name__ == "__main__":
    main()
