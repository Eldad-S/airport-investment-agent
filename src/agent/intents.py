"""Structured representation of a user question. Produced by the rule-based router or by the
optional LLM parser; consumed by the orchestrator. Pydantic validates both paths."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

IntentName = Literal[
    "rank",            # which airports are the best opportunities (optionally in a region)
    "compare",         # compare 2+ airports (optionally focused on congestion / demand)
    "congestion",      # congestion evidence for one airport
    "long_haul",       # long-haul share of departures
    "demand",          # demand / unmet-demand evidence
    "profile",         # general metrics for one airport
    "sensitivity",     # weight changes / robustness / which assumptions matter
    "why_rank",        # why airport A ranks above / below B
    "sources",         # data sources and provenance
    "data_needed",     # what extra data would validate an investment
    "methodology",     # how scores are calculated
    "help",            # greetings / what can you do
    "unsupported_aviation",  # aviation topics we have no data for (weather, fares, flight status)
    "off_topic",       # unrelated to airports / investment
    "clarify",         # relevant but missing an airport we cannot infer
]


class ParsedQuery(BaseModel):
    intent: IntentName
    airports: list[str] = Field(default_factory=list, description="IATA codes, in order mentioned")
    region: Optional[str] = Field(None, description="Region name (e.g. 'New England') or 2-letter state code")
    focus: Optional[Literal["congestion", "demand", "long_haul", "general"]] = None
    top_n: Optional[int] = None
    threshold_miles: Optional[float] = None
    demand_weight: Optional[float] = None
    capacity_weight: Optional[float] = None
    operations_weight: Optional[float] = None
    weight_mode: Optional[Literal["absolute", "delta"]] = None
    unknown_codes: list[str] = Field(default_factory=list)
    off_topic_subject: Optional[str] = None
    used_context: bool = False
    parser: Literal["rules", "llm"] = "rules"

    def weight_changes(self) -> dict[str, float]:
        return {k: v for k, v in (("demand", self.demand_weight), ("capacity_pressure", self.capacity_weight),
                                  ("operations", self.operations_weight)) if v is not None}


class ConversationContext(BaseModel):
    """What follow-up questions may refer to. Kept explicit so it cannot drift silently."""
    last_intent: Optional[str] = None
    last_airports: list[str] = Field(default_factory=list)
    last_region: Optional[str] = None
    last_focus: Optional[str] = None
