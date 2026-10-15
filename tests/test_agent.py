"""Router unit tests + evaluation cases for the agent (LLM disabled)."""
import re

import pytest

from src.agent.intents import ConversationContext, ParsedQuery
from src.agent.router import Router

CODES = {"LAX", "SNA", "SFO", "ANC", "BOS", "BDL", "JFK", "SAN", "DEN", "MCO", "SUN", "DAY", "SEA", "ORD"}


@pytest.fixture
def router():
    return Router(CODES, {"anchorage": "ANC", "orlando": "MCO", "denver": "DEN"})


@pytest.mark.parametrize("text,intent,airports", [
    ("compare congestion LAX SNA", "compare", ["LAX", "SNA"]),
    ("Compare congestion levels at Los Angeles International Airport (LAX) and John Wayne Airport in Santa Ana (SNA).", "compare", ["LAX", "SNA"]),
    ("lax vs sna", "compare", ["LAX", "SNA"]),
    ("how jammed is sfo", "congestion", ["SFO"]),
    ("sfo delays bad?", "congestion", ["SFO"]),
    ("What percentage of flights departing Anchorage airport are long-haul flights?", "long_haul", ["ANC"]),
    ("% long haul out of anchorge", "long_haul", ["ANC"]),  # typo
    ("unmet demand sfo", "demand", ["SFO"]),
    ("What is the unmet flight demand at San Francisco International Airport (SFO), and what might explain it?", "demand", ["SFO"]),
    ("Which airports offer the most promising opportunities according to the investment scoring model?", "rank", []),
    ("What additional data would be needed to validate a potential investment?", "data_needed", []),
    ("why did BOS rank above BDL", "why_rank", ["BOS", "BDL"]),
    ("is it worth investing in denver or nah", "profile", ["DEN"]),
    ("what's the deal with orlando", "profile", ["MCO"]),
    ("sna", "profile", ["SNA"]),
    ("how is the score calculated", "methodology", []),
    ("which data sources do you use", "sources", []),
])
def test_router_intents(router, text, intent, airports):
    q = router.parse(text)
    assert q.intent == intent, q
    assert q.airports == airports


def test_router_region_and_top_n(router):
    q = router.parse("Which airports in New England are strong candidates for terminal expansion?")
    assert (q.intent, q.region) == ("rank", "New England")
    q = router.parse("top 5 in texas")
    assert (q.intent, q.region, q.top_n) == ("rank", "TX", 5)


def test_common_words_are_not_airport_codes(router):
    # 'sun' and 'day' are real airport codes (SUN, DAY) but common words in lower case
    q = router.parse("which airports have the best day in the sun for investors")
    assert q.airports == []
    assert router.parse("compare SUN and DAY").airports == ["SUN", "DAY"]


def test_unknown_codes_are_reported(router):
    q = router.parse("LHR vs JFK")
    assert "LHR" in q.unknown_codes and q.airports == ["JFK"]


def test_weight_what_if_parsing(router):
    q = router.parse("what if capacity weight is 50%")
    assert q.intent == "sensitivity" and q.capacity_weight == 0.5 and q.weight_mode == "absolute"
    q = router.parse("what happens if i bump capacity weight")
    assert q.intent == "sensitivity" and q.capacity_weight == pytest.approx(0.15) and q.weight_mode == "delta"
    assert router.parse("which assumptions matter most").intent == "sensitivity"


@pytest.mark.parametrize("text,intent", [
    ("tell me a joke", "off_topic"),
    ("who won the super bowl", "off_topic"),
    ("what's the capital of France", "off_topic"),
    ("is the weather nice at LAX?", "unsupported_aviation"),
    ("cheapest flight to BOS", "unsupported_aviation"),
    ("hi", "help"),
    ("how congested is it", "clarify"),  # no airport and no context
    ("how's the airport near Disney doing", "clarify"),  # on-topic, airport not recognized
    ("how busy is the airport by the space center", "clarify"),
    ("tell me a joke about airplanes", "off_topic"),  # aviation word, but clearly unrelated
])
def test_unsupported_and_ambiguous(router, text, intent):
    assert router.parse(text).intent == intent


def test_followups_use_context(router):
    ctx = ConversationContext(last_intent="congestion", last_airports=["SFO"], last_focus="congestion")
    q = router.parse("and BOS?", ctx)
    assert (q.intent, q.airports, q.used_context) == ("congestion", ["BOS"], True)
    q = router.parse("show the evidence behind the congestion score", ctx)
    assert (q.intent, q.airports) == ("congestion", ["SFO"])
    ctx = ConversationContext(last_intent="compare", last_airports=["LAX", "SNA"], last_focus="congestion")
    q = router.parse("compare both airports over the same period", ctx)
    assert (q.intent, q.airports) == ("compare", ["LAX", "SNA"])
    q = router.parse("what about SAN", ctx)
    assert (q.intent, q.airports) == ("compare", ["LAX", "SNA", "SAN"])
    ctx = ConversationContext(last_intent="rank", last_region="New England")
    q = router.parse("what about the midwest", ctx)
    assert (q.intent, q.region) == ("rank", "Midwest")


# ---------------------------------------------------------------------------------------
# Evaluation cases on the real snapshot (expected *behaviour*, not hard-coded numbers)
# ---------------------------------------------------------------------------------------
SUBSTANTIVE = ["**Period:**", "**Sources:**", "**Confidence:**", "**Caveats**", "Bottom line"]


def _ask(agent, text, ctx=None):
    r, ctx = agent.ask(text, ctx or ConversationContext())
    return r, ctx


@pytest.mark.parametrize("question", [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare congestion levels at Los Angeles International Airport (LAX) and John Wayne Airport in Santa Ana (SNA).",
    "What percentage of flights departing Anchorage airport are long-haul flights?",
    "What is the unmet flight demand at San Francisco International Airport (SFO), and what might explain it?",
    "Which airports offer the most promising opportunities according to the investment scoring model?",
])
def test_required_questions_have_full_answer_structure(real_agent, question):
    r, _ = _ask(real_agent, question)
    for marker in SUBSTANTIVE:
        assert marker in r.text, f"missing {marker!r}"
    assert r.tool_calls, "answer must come from a deterministic tool"


def test_eval_sfo_has_no_numeric_unmet_demand(real_agent):
    r, _ = _ask(real_agent, "What is the unmet flight demand at SFO?")
    assert "cannot be quantified" in r.text.lower() or "cannot be quantified" in r.text
    # no sentence like "unmet demand of 1.2M" / "unmet demand is 15%"
    assert not re.search(r"unmet (flight )?demand (of|is|=|estimated at)\s*[\d~]", r.text, re.I)
    assert "T-100" in r.text  # explains what data would be needed


def test_eval_anc_long_haul_reports_numerator_denominator(real_agent):
    r, _ = _ask(real_agent, "What percentage of flights departing Anchorage are long-haul?")
    assert re.search(r"\*\*\d+\.\d%\*\*", r.text)
    assert re.search(r"[\d,]+ long-haul departures / [\d,]+ eligible departures", r.text)
    assert "2,485" in r.text and "cancelled" in r.text.lower()


def test_eval_new_england_only_lists_new_england(real_agent):
    r, _ = _ask(real_agent, "top airports in new england")
    ranking = r.tables[0][1]
    assert set(ranking["State"]) <= {"CT", "ME", "MA", "NH", "RI", "VT"}
    assert "BOS" in ranking.index


def test_eval_compare_uses_same_period(real_agent):
    r, _ = _ask(real_agent, "compare congestion LAX SNA")
    assert "identical period" in r.text
    assert r.tables[0][1].columns.tolist() == ["LAX", "SNA"]


def test_eval_off_topic_is_declined(real_agent):
    r, _ = _ask(real_agent, "what's a good pasta recipe")
    assert "outside what I can help with" in r.text and not r.tool_calls


def test_eval_score_is_not_called_roi(real_agent):
    r, _ = _ask(real_agent, "best airports to invest in")
    assert "not an ROI" in r.text


def test_eval_conversation_followups(real_agent):
    r, ctx = _ask(real_agent, "how jammed is sfo")
    r2, ctx = _ask(real_agent, "and BOS?", ctx)
    assert r2.parsed.intent == "congestion" and r2.parsed.airports == ["BOS"]
    r3, ctx = _ask(real_agent, "why did BOS rank above BDL", ctx)
    assert "ranks above" in r3.text or "ranks" in r3.text


def test_eval_what_if_does_not_change_base_weights(real_agent):
    before = dict(real_agent.engine.weights)
    r, _ = _ask(real_agent, "what if capacity weight is 60%")
    assert "Capacity pressure 60%" in r.text
    assert real_agent.engine.weights == before


def test_llm_failure_falls_back_to_rules(real_agent):
    class FailingLLM:
        def parse(self, *a, **k):
            return None

    real_agent.llm = FailingLLM()
    try:
        r, _ = _ask(real_agent, "compare congestion LAX SNA")
        assert r.parsed.parser == "rules" and r.parsed.intent == "compare"
    finally:
        real_agent.llm = None


def test_llm_output_cannot_inject_numbers(real_agent):
    class StubLLM:  # an LLM can only choose the tool; numbers still come from the engine
        def parse(self, *a, **k):
            return ParsedQuery(intent="congestion", airports=["SFO"], parser="llm")

    real_agent.llm = StubLLM()
    try:
        r, _ = _ask(real_agent, "anything")
        rules, _ = _ask_rules(real_agent, "how congested is SFO")
        assert r.parsed.parser == "llm"
        assert r.text.split("**Bottom line:**")[1][:200] == rules.text.split("**Bottom line:**")[1][:200]
    finally:
        real_agent.llm = None


def _ask_rules(agent, text):
    saved, agent.llm = agent.llm, None
    try:
        return agent.ask(text, ConversationContext())
    finally:
        agent.llm = saved
