"""Optional Claude-based question understanding.

The LLM is used ONLY to turn a free-form question into a validated `ParsedQuery` (which tool
to run, which airports, region, weights). It never sees or produces the numbers - those come
from the deterministic tools. Any failure (no key, network, refusal, invalid output) falls
back to the rule-based router, so the app always works without an API key.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.agent.intents import ConversationContext, IntentName, ParsedQuery
from src.agent.router import normalize_region
from src.config import ANTHROPIC_API_KEY, LLM_ENABLED, LLM_MODEL

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You translate questions from airport-investment analysts into a structured
query for a deterministic analytics engine covering U.S. airports (BTS on-time data and FAA
passenger boardings, calendar years 2024-2025). You never answer the question yourself and
never produce numbers about airports.

Intents:
- rank: which airports are the best investment / expansion opportunities (optionally in a region or state)
- compare: compare two or more airports (set focus to congestion / demand / general)
- congestion: congestion / delay / capacity-pressure evidence for an airport
- long_haul: share of long-haul departures (threshold_miles only if the user states one)
- demand: demand or unmet-demand evidence for an airport
- profile: general metrics for an airport
- sensitivity: what if weights change, robustness, which assumptions matter most
- why_rank: why airport A ranks above/below airport B (or why an airport ranks where it does)
- sources: which data sources support a conclusion
- data_needed: what additional data would validate an investment
- methodology: how the scores are calculated
- help: greetings, what can you do
- unsupported_aviation: aviation topics without data here (weather, fares, live flight status, TSA lines)
- off_topic: anything unrelated to airports or airport investment
- clarify: on-topic but the airport cannot be determined even from context

Rules:
- Analysts write informally ("compare congestion LAX SNA", "how jammed is sfo", "top 5 new england").
- Map airport names, cities and nicknames to U.S. IATA codes (e.g. "John Wayne" -> SNA,
  "Anchorage" -> ANC, "Logan" -> BOS). A city with several airports maps to its main airport.
- region: one of New England, Mid-Atlantic, Southeast, Midwest, Southwest, Mountain West,
  West Coast, Alaska & Hawaii, or a 2-letter U.S. state code.
- Weights: demand_weight / capacity_weight / operations_weight. Use weight_mode "absolute"
  with fractions (50% -> 0.5) when the user gives numbers, or "delta" with +0.15 / -0.15 for
  "more" / "less".
- Use the conversation context for follow-ups ("what about SNA?", "why?", "show the evidence").
- Non-U.S. airports: put the code in airports anyway; the engine will report it as unknown."""


class LLMIntent(BaseModel):
    intent: IntentName
    airports: list[str] = Field(default_factory=list)
    region: Optional[str] = None
    focus: Optional[Literal["congestion", "demand", "long_haul", "general"]] = None
    top_n: Optional[int] = None
    threshold_miles: Optional[float] = None
    demand_weight: Optional[float] = None
    capacity_weight: Optional[float] = None
    operations_weight: Optional[float] = None
    weight_mode: Optional[Literal["absolute", "delta"]] = None
    off_topic_subject: Optional[str] = None


class LLMIntentParser:
    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = LLM_MODEL):
        import anthropic  # imported lazily so the app runs even if the package is absent

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=30.0, max_retries=1)
        self.model = model

    def parse(self, text: str, ctx: ConversationContext, known_codes: set[str]) -> ParsedQuery | None:
        context = (f"Conversation context: last_intent={ctx.last_intent}, last_airports={ctx.last_airports}, "
                   f"last_region={ctx.last_region}")
        try:
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"{context}\n\nQuestion: {text}"}],
                output_format=LLMIntent,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except self._anthropic.APIConnectionError as e:
            log.warning("LLM unreachable, using rule-based router: %s", e)
            return None
        except self._anthropic.APIStatusError as e:
            log.warning("LLM API error %s, using rule-based router: %s", e.status_code, e)
            return None
        except Exception:  # noqa: BLE001 - the demo must never break on the optional LLM path
            log.exception("Unexpected LLM failure, using rule-based router")
            return None
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            log.warning("LLM returned no usable parse (stop_reason=%s)", resp.stop_reason)
            return None
        out: LLMIntent = resp.parsed_output
        codes = [c.strip().upper() for c in out.airports]
        known = [c for c in codes if c in known_codes]
        unknown = [c for c in codes if c not in known_codes]
        # The LLM may phrase regions freely ("Northeast", "California"); map them with the same
        # normalizer the rule-based router uses, falling back to the question text itself.
        region = normalize_region(out.region) or (normalize_region(text) if out.region else None)
        unmapped_region = out.region if out.region and region is None else None
        intent = out.intent
        if intent in ("compare", "congestion", "long_haul", "demand", "profile", "why_rank") and not known and not unknown:
            intent = "clarify"
        return ParsedQuery(
            intent=intent, airports=known, region=region, unmapped_region=unmapped_region,
            focus=out.focus, top_n=out.top_n,
            threshold_miles=out.threshold_miles, demand_weight=out.demand_weight,
            capacity_weight=out.capacity_weight, operations_weight=out.operations_weight,
            weight_mode=out.weight_mode, unknown_codes=unknown, off_topic_subject=out.off_topic_subject,
            used_context=bool(ctx.last_airports) and set(known) <= set(ctx.last_airports) and bool(known)
            and not any(c.lower() in text.lower() for c in known),
            parser="llm",
        )


def make_llm_parser() -> LLMIntentParser | None:
    if not LLM_ENABLED:
        return None
    try:
        return LLMIntentParser()
    except ImportError:
        log.warning("anthropic package not installed; LLM disabled")
        return None
