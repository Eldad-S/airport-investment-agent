"""Region handling on the LLM path. The Anthropic client is mocked - no network calls."""
from types import SimpleNamespace

import pytest

from src.agent.intents import ConversationContext, ParsedQuery
from src.agent.llm import LLMIntent, LLMIntentParser
from src.agent.router import normalize_region


@pytest.mark.parametrize("raw,expected", [
    ("New England", "New England"), ("new england", "New England"), ("Northeast", "New England"),
    ("California", "CA"), ("california", "CA"), ("CA", "CA"), ("tx", "TX"), ("Texas", "TX"),
    ("Washington", "WA"), ("New York", "NY"), ("Pacific Northwest", "West Coast"), ("west coast", "West Coast"),
    ("Midwest", "Midwest"), ("Sun Belt", None), ("Narnia", None), ("", None), (None, None),
])
def test_normalize_region(raw, expected):
    assert normalize_region(raw) == expected


def _parser_returning(out: LLMIntent) -> LLMIntentParser:
    """Build a parser whose Anthropic client returns `out` without touching the network."""
    import anthropic

    p = LLMIntentParser.__new__(LLMIntentParser)
    p._anthropic = anthropic
    p.model = "test-model"
    resp = SimpleNamespace(stop_reason="end_turn", parsed_output=out)
    p.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: resp)))
    return p


@pytest.mark.parametrize("llm_region,expected", [("California", "CA"), ("Northeast", "New England"),
                                                 ("new england", "New England")])
def test_llm_region_is_normalized(llm_region, expected):
    p = _parser_returning(LLMIntent(intent="rank", region=llm_region))
    q = p.parse("top airports", ConversationContext(), {"LAX"})
    assert q.region == expected and q.unmapped_region is None and q.parser == "llm"


def test_llm_unmappable_region_is_flagged():
    p = _parser_returning(LLMIntent(intent="rank", region="Sun Belt"))
    q = p.parse("best sun belt airports", ConversationContext(), {"LAX"})
    assert q.region is None and q.unmapped_region == "Sun Belt"


def test_unmapped_region_answer_says_national(real_agent):
    class StubLLM:
        def parse(self, *a, **k):
            return ParsedQuery(intent="rank", unmapped_region="Sun Belt", parser="llm")

    real_agent.llm = StubLLM()
    try:
        r, ctx = real_agent.ask("best sun belt airports", ConversationContext())
        assert "couldn't map \"Sun Belt\"" in r.text and "national" in r.text
        assert "in the U.S." in r.text and "in Sun Belt" not in r.text
        assert ctx.last_region is None
    finally:
        real_agent.llm = None


def test_invalid_region_never_mislabels_national_list(real_agent):
    class StubLLM:  # even if an un-normalized region slips through, it must not be used as a label
        def parse(self, *a, **k):
            return ParsedQuery(intent="rank", region="Atlantis", parser="llm")

    real_agent.llm = StubLLM()
    try:
        r, _ = real_agent.ask("top airports in atlantis", ConversationContext())
        assert "couldn't map \"Atlantis\"" in r.text and "in Atlantis" not in r.text
    finally:
        real_agent.llm = None
