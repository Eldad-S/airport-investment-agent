"""Verify the optional Claude integration with a single small request.

    python scripts/check_llm.py "compare congestion LAX SNA"

Prints the structured query Claude produced (or why it fell back to the rule-based router).
Makes one API call billed to the configured ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.intents import ConversationContext  # noqa: E402
from src.agent.llm import make_llm_parser  # noqa: E402
from src.config import LLM_ENABLED  # noqa: E402
from src.data_sources.repository import load_dataset  # noqa: E402
from src.agent.tools import AnalyticsEngine  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not LLM_ENABLED:
        sys.exit("LLM disabled: set ANTHROPIC_API_KEY in .env (and DISABLE_LLM=0).")
    question = " ".join(sys.argv[1:]) or "compare congestion LAX SNA"
    parser = make_llm_parser()
    codes = AnalyticsEngine(load_dataset()).known_codes()
    q = parser.parse(question, ConversationContext(), codes)
    print("FAILED - the app would fall back to the rule-based router." if q is None else q.model_dump_json(indent=2))
