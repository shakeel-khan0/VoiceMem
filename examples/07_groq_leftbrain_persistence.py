"""Groq fact extraction + local E5 persistence smoke test.

Run from the repository root in two processes:

    python examples/07_groq_leftbrain_persistence.py ingest
    python examples/07_groq_leftbrain_persistence.py search
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep the standalone script runnable from a source clone without installing it.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from voicemem import VoiceMem


MEMORY = "My name is Ali Khan. I run a dental clinic and we miss calls after business hours."
QUERY = "What business does Ali run and what problem does he have?"
MEMORY_ROOT = Path(__file__).resolve().parent / "groq_leftbrain_memory"


def make_voicemem() -> VoiceMem:
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is required")
    return VoiceMem.from_config({
        "mode": "leftbrain_only",
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
        result = vm.ingest(MEMORY)
        memory_ids = result.get("memory_ids", [])
        if not memory_ids:
            raise SystemExit("FAIL: Groq extracted no memories")
        print(f"PASS: persisted {len(memory_ids)} memories")
        return

    result = vm.search(QUERY, top_k=5)
    memories = result.result_leftbrain or []
    rendered = "\n".join(str(item) for item in memories)
    print(rendered)
    lowered = rendered.lower()
    if "dental" not in lowered or "call" not in lowered:
        raise SystemExit("FAIL: persisted dental-clinic/call facts were not retrieved")
    print("PASS: retrieved persisted Ali Khan business and missed-call memories")


if __name__ == "__main__":
    main()
