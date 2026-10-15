"""Deterministic natural-language router. Works without any LLM.

Designed for informal analyst phrasing: "compare congestion LAX SNA", "how jammed is sfo",
"top 5 new england", "% long haul out of anchorage", "what if capacity weight 50%".
"""
from __future__ import annotations

import difflib
import re

from src.agent.intents import ConversationContext, ParsedQuery
from src.regions import (AIRPORT_ALIASES, AMBIGUOUS_LOWERCASE, METRO_GROUPS, REGION_ALIASES,
                         REGIONS, STATE_NAMES)

# Upper-case 3-letter words that are not airport codes.
NON_AIRPORT_ACRONYMS = {"ROI", "NPV", "IRR", "USA", "FAA", "BTS", "NAS", "LOL", "WTF", "CEO", "API",
                        "TSA", "ATC", "GDP", "KPI", "AND", "THE", "FOR", "YOU", "ARE", "OMG", "IDK",
                        "PLS", "LLM", "CSV", "EPS", "IPO", "FYI", "ETA", "TBD", "USD", "CAP", "OPS"}

KW = {
    "data_needed": [r"additional data", r"more data", r"what data", r"which data (would|do|should)",
                    r"data (would|do) (we|i|you) need", r"need(ed)? to validate", r"validate",
                    r"due diligence", r"missing data", r"improve (this|the) (assessment|analysis)",
                    r"what else (do|would) (we|i|you) need", r"data gaps?"],
    "sources": [r"\bsources?\b", r"where (does|did|do) (the|this|that|your) data", r"provenance",
                r"\bcite\b", r"citations?", r"where.*numbers come from", r"data from\b"],
    "methodology": [r"methodolog", r"how (do|does|is|are) (you|the|it) (calculat|comput|scor|rank|work)",
                    r"formula", r"how.*score.*(work|calculated|computed)", r"what does the score mean",
                    r"explain (the )?(scor|model)"],
    "sensitivity": [r"sensitiv", r"robust", r"what if", r"\bweights?\b", r"assumptions?",
                    r"what happens if", r"more emphasis", r"prioriti[sz]e"],
    "why_rank": [r"\bwhy\b", r"how come", r"rank(ed|s)? (above|higher|over|below|lower)",
                 r"outrank", r"\bbeat(s)?\b", r"ahead of"],
    "long_haul": [r"long[\s-]?haul", r"long flights", r"long[\s-]distance", r"short[\s-]?haul",
                  r"\bfar\b.*flights", r"flights.*\bfar\b", r"how far"],
    "demand": [r"unmet", r"pent[\s-]?up", r"suppress", r"latent", r"\bdemand\b", r"underserved",
               r"under[\s-]served", r"spill"],
    "compare": [r"compar", r"\bvs\.?\b", r"\bv\.?\b", r"versus", r"against", r"\bdiff(erence)?s?\b",
                r"side by side", r"better than", r"worse than", r"more congested than", r"\bboth\b"],
    "congestion": [r"congest", r"\bjam", r"crowd", r"\bbusy\b", r"busiest", r"delay", r"\blate\b",
                   r"bottleneck", r"packed", r"gridlock", r"taxi", r"slammed", r"\bmess\b", r"chaos",
                   r"on[\s-]?time", r"capacity", r"constrain", r"pressure", r"maxed"],
    "rank": [r"\brank", r"\btop\b", r"\bbest\b", r"promising", r"opportunit", r"\binvest", r"candidate",
             r"\bscor", r"where should", r"which airports", r"shortlist", r"expan", r"renovat",
             r"terminal", r"upgrade", r"worth", r"leaderboard", r"\bpicks?\b", r"strong", r"winners?",
             r"prioriti", r"profitab", r"\bdeals?\b"],
    "profile": [r"tell me about", r"\binfo\b", r"\bstats\b", r"numbers on", r"overview", r"profile",
                r"how('?s| is) .* doing", r"\bdetails?\b", r"metrics", r"snapshot", r"lowdown", r"rundown"],
    "help": [r"^(hi|hey|hello|yo|sup|hiya|howdy|shalom)\b", r"what can you do", r"\bhelp\b",
             r"how do i use", r"who are you", r"^thanks?", r"^thank you", r"^ty\b"],
}
UNSUPPORTED_AVIATION = {
    "weather": r"weather|forecast|rain|snow|storm|temperature",
    "fares / ticket prices": r"\bfares?\b|ticket|cheap(est)? flight|price of (a )?flight|book(ing)? a flight|\bbook\b",
    "live flight status": r"flight status|is my flight|gate (number|change)|arrival time of|track(ing)? (a |my )?flight|right now|live\b|real[\s-]?time",
    "security wait times": r"tsa (line|wait)|security (line|wait)",
    "parking / lounges / food": r"parking|lounge|restaurant|food court|wifi",
}
OFF_TOPIC_HINTS = r"joke|recipe|poem|song|movie|football|soccer|nba|crypto|bitcoin|stock price|president|capital of|meaning of life|girlfriend|boyfriend|homework|translate|who won"

FOLLOWUP_PRONOUNS = r"\b(it|its|there|that airport|this airport|them|those|both|these|same|that one|this one)\b"


def _norm(text: str) -> str:
    t = text.replace("’", "'").replace("&", " and ").lower()
    t = re.sub(r"[“”\"?!,;:()\[\]]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_region(value: str | None) -> str | None:
    """Map a free-form region ("Northeast", "california", "CA", "Pacific Northwest") to a region
    name in REGIONS or a 2-letter state code. Returns None when it cannot be mapped."""
    if not value or not value.strip():
        return None
    v = value.strip()
    if v in REGIONS:
        return v
    for name in REGIONS:
        if name.lower() == v.lower():
            return name
    if v.upper() in STATE_NAMES and len(v) == 2:
        return v.upper()
    return Router.extract_region(v)


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


class Router:
    def __init__(self, known_codes: set[str], city_to_code: dict[str, str] | None = None):
        self.known = {c.upper() for c in known_codes}
        self.city_to_code = {k: v for k, v in (city_to_code or {}).items() if v in self.known}

    # ---- entity extraction --------------------------------------------------------------
    def extract_airports(self, raw: str) -> tuple[list[str], list[str]]:
        low = " " + _norm(raw) + " "
        found: list[tuple[int, str]] = []
        unknown: list[str] = []

        def take(phrase: str, codes: tuple[str, ...]):
            nonlocal low
            for m in re.finditer(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", low):
                for c in codes:
                    if c in self.known:
                        found.append((m.start(), c))
            low = re.sub(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lambda m: " " * len(m.group()), low)

        for phrase, codes in sorted(METRO_GROUPS.items(), key=lambda x: -len(x[0])):
            take(phrase, codes)
        aliases = {**{k: v for k, v in self.city_to_code.items()}, **AIRPORT_ALIASES}
        for phrase, code in sorted(aliases.items(), key=lambda x: -len(x[0])):
            if len(phrase) <= 3:
                continue  # short aliases handled as tokens below (case-sensitive rules)
            take(phrase, (code,))

        # 3-character tokens, using the ORIGINAL casing to resolve ambiguous English words.
        for m in re.finditer(r"(?<![A-Za-z0-9])([A-Za-z0-9]{2,3})(?![A-Za-z0-9])", raw):
            tok = m.group(1)
            pos = m.start()
            if low[pos + 1: pos + 1 + len(tok)].strip() == "":
                continue  # already consumed as part of a longer phrase
            lower = tok.lower()
            if len(tok) == 2 or lower in ("la", "sf", "dc"):
                alias = AIRPORT_ALIASES.get(lower)
                if alias and (lower != "la" or not re.search(r"\bla (guardia|airports?)\b", raw.lower())):
                    if alias in self.known:
                        found.append((pos, alias))
                continue
            up = tok.upper()
            if up in self.known:
                if tok.isupper() or lower not in AMBIGUOUS_LOWERCASE:
                    found.append((pos, up))
            elif tok.isupper() and tok.isalpha() and up not in NON_AIRPORT_ACRONYMS:
                unknown.append(up)

        # Fuzzy match for typos of city names ("anchorge", "sanfransisco").
        if not found:
            for w in re.findall(r"[a-z]{6,}", _norm(raw)):
                close = difflib.get_close_matches(w, list(aliases), n=1, cutoff=0.85)
                if close and aliases[close[0]] in self.known:
                    found.append((0, aliases[close[0]]))

        ordered: list[str] = []
        for _, c in sorted(found, key=lambda x: x[0]):
            if c not in ordered:
                ordered.append(c)
        return ordered, unknown

    @staticmethod
    def extract_region(raw: str) -> str | None:
        low = " " + _norm(raw) + " "
        for phrase, region in sorted(REGION_ALIASES.items(), key=lambda x: -len(x[0])):
            if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", low):
                return region
        for name in REGIONS:
            if name.lower() in low:
                return name
        for code, name in STATE_NAMES.items():
            if re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", low):
                return code
        for code, name in (("NY", "new york"), ("WA", "washington")):
            if low.strip() == name:  # bare state name, e.g. an LLM returning "Washington"
                return code
        for name in ("new york", "washington"):
            if re.search(rf"\b{name}\b (state|airports)", low):
                return "NY" if name == "new york" else "WA"
        m = re.search(r"\b(?:in|across|within)\s+([A-Z]{2})\b", raw)
        if m and m.group(1) in STATE_NAMES:
            return m.group(1)
        return None

    @staticmethod
    def extract_numbers(text: str) -> tuple[int | None, float | None]:
        low = _norm(text)
        top = None
        m = re.search(r"\btop\s*(\d{1,2})\b", low) or re.search(r"\b(\d{1,2})\s*(best|top|most promising|airports)\b", low)
        if m:
            top = int(m.group(1))
        thr = None
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(mi|miles?|km|kilometers?)\b", low)
        if m:
            v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
            thr = v * 0.621371 if m.group(3).startswith("k") else v
        return top, thr

    @staticmethod
    def extract_weights(text: str) -> tuple[dict[str, float], str | None]:
        low = _norm(text)
        comps = {
            "capacity_weight": r"(capacity|congestion|pressure|constraint)",
            "demand_weight": r"(demand|passenger|pax|growth)",
            "operations_weight": r"(operations|operational|ops|flight activity|flights)",
        }
        out: dict[str, float] = {}
        mode = None
        for field, pat in comps.items():
            m = (re.search(pat + r"[a-z ]{0,20}?(?:weight)?[a-z ]{0,10}?(?:to|=|at|of|is)?\s*(\d{1,3})\s*%", low)
                 or re.search(r"(\d{1,3})\s*%[a-z ]{0,15}?(?:on|for|to)?\s*" + pat, low))
            if m:
                num = next(g for g in m.groups() if g and g.isdigit())
                out[field] = int(num) / 100
                mode = "absolute"
                continue
            if re.search(r"(increase|more|bump|raise|boost|higher|up|double|heavier|emphasi[sz]e|prioriti[sz]e)[a-z ]{0,25}" + pat, low) \
                    or re.search(pat + r"[a-z ]{0,15}(weight)?[a-z ]{0,5}(up|higher|increased|more)", low):
                out[field] = 0.15
                mode = mode or "delta"
            elif re.search(r"(decrease|less|lower|reduce|cut|drop|ignore|zero)[a-z ]{0,25}" + pat, low):
                out[field] = -0.15
                mode = mode or "delta"
        if mode == "delta" and not re.search(r"weight|what if|emphasi|prioriti|matter|if i|if we|happens", low):
            return {}, None
        return out, mode

    # ---- main -------------------------------------------------------------------------
    def parse(self, text: str, ctx: ConversationContext | None = None) -> ParsedQuery:
        ctx = ctx or ConversationContext()
        low = _norm(text)
        airports, unknown = self.extract_airports(text)
        region = self.extract_region(text)
        top_n, threshold = self.extract_numbers(text)
        weights, wmode = self.extract_weights(text)
        has = {k: _any(v, low) for k, v in KW.items()}
        used_ctx = False

        unsupported = next((name for name, pat in UNSUPPORTED_AVIATION.items() if re.search(pat, low)), None)
        aviation_word = re.search(r"airport|flight|airline|aviation|runway|terminal|passenger|plane|gate|hub", low)

        intent: str | None = None
        focus = None
        followup_prefix = bool(re.match(r"^(and|what about|how about|same for|ok and|now|also|what of)\b", low))
        airport_intents = ("congestion", "long_haul", "demand", "profile")
        other_intent_words = any(has[k] for k in ("data_needed", "sources", "methodology", "long_haul",
                                                  "demand", "compare", "congestion", "why_rank"))
        evidence_req = bool(re.search(r"evidence|behind|breakdown|show me|drill", low))
        if followup_prefix and airports and ctx.last_intent in airport_intents and not other_intent_words:
            intent, focus, used_ctx = ctx.last_intent, ctx.last_focus, True
        elif followup_prefix and airports and ctx.last_intent == "compare" and not other_intent_words:
            intent, focus, used_ctx = "compare", ctx.last_focus, True
            airports = list(dict.fromkeys(ctx.last_airports + airports))
        elif has["data_needed"]:
            intent = "data_needed"
        elif has["sources"] and not airports:
            intent = "sources"
        elif has["methodology"] and not has["compare"]:
            intent = "methodology"
        elif weights or (has["sensitivity"] and re.search(r"weight|sensitiv|robust|assumption|what if|happens if", low)):
            intent = "sensitivity"
        elif has["why_rank"] and (len(airports) >= 2 or re.search(r"rank|score|above|below|higher|lower|ahead|beat", low)):
            intent = "why_rank"
        elif has["long_haul"]:
            intent, focus = "long_haul", "long_haul"
        elif has["demand"] and not has["rank"]:
            intent, focus = "demand", "demand"
        elif has["compare"] or len(airports) >= 2:
            intent = "compare"
            focus = "congestion" if has["congestion"] else "demand" if has["demand"] else "general"
        elif has["congestion"] and (airports or not has["rank"] or (evidence_req and ctx.last_airports)):
            intent, focus = "congestion", "congestion"
        elif (has["rank"] or region) and not airports:
            intent = "rank"
        elif has["rank"] and len(airports) == 1:
            intent = "profile"  # "is it worth investing in denver" -> that airport's score & rank
        elif unsupported and (airports or aviation_word):
            intent = "unsupported_aviation"
        elif airports and (has["profile"] or len(low.split()) <= 6):
            intent = "profile"
        elif has["help"]:
            intent = "help"

        # --- Follow-ups: "what about SNA?", "and BOS", "show the evidence", "why?" -------------
        is_short_followup = bool(re.match(r"^(and|what about|how about|same for|ok and|now|also)\b", low)) \
            or bool(re.search(FOLLOWUP_PRONOUNS, low))
        if intent is None and is_short_followup and ctx.last_intent:
            if airports and ctx.last_intent in ("congestion", "long_haul", "demand", "profile", "compare"):
                intent, focus, used_ctx = ctx.last_intent, ctx.last_focus, True
                if intent == "compare":
                    airports = list(dict.fromkeys(ctx.last_airports + airports))
            elif region and ctx.last_intent == "rank":
                intent, used_ctx = "rank", True
        if intent is None and unsupported:
            intent = "unsupported_aviation" if (airports or aviation_word) else "off_topic"
        if intent is None and (airports or region):
            intent = "profile" if airports else "rank"
        if intent is None and has["why_rank"] and ctx.last_intent:
            intent = "why_rank"
        if intent is None and ("evidence" in low or "show me" in low or "details" in low) and ctx.last_airports:
            intent, used_ctx = ctx.last_intent if ctx.last_intent in ("congestion", "demand", "long_haul") else "profile", True
        if intent is None:
            # On-topic wording ("the airport near Disney") but no airport recognized: ask which one
            # instead of declining. Clearly unrelated requests ("a joke about planes") stay off-topic.
            intent = "clarify" if aviation_word and not re.search(OFF_TOPIC_HINTS, low) else "off_topic"

        # --- Fill missing airports from context ------------------------------------------
        needs_airport = intent in ("compare", "congestion", "long_haul", "demand", "profile", "why_rank")
        if needs_airport and not airports and ctx.last_airports:
            airports, used_ctx = list(ctx.last_airports), True
        if intent == "compare" and len(airports) == 1 and ctx.last_airports and \
                (re.search(FOLLOWUP_PRONOUNS, low) or re.search(r"\bwith\b|\bto\b", low)):
            airports, used_ctx = list(dict.fromkeys(ctx.last_airports + airports)), True
        if intent == "rank" and not region and ctx.last_intent == "rank" and is_short_followup:
            region, used_ctx = ctx.last_region, True
        if intent == "sources" and focus is None:
            focus = ctx.last_focus
        if needs_airport and not airports:
            intent = "clarify"
        if intent == "compare" and len(airports) == 1:
            intent = {"congestion": "congestion", "demand": "demand"}.get(focus or "", "profile")

        off_subject = None
        if intent == "off_topic":
            m = re.search(OFF_TOPIC_HINTS, low)
            off_subject = m.group(0) if m else None
        if intent == "unsupported_aviation":
            off_subject = unsupported

        return ParsedQuery(
            intent=intent, airports=airports, region=region, focus=focus, top_n=top_n,
            threshold_miles=threshold, weight_mode=wmode, unknown_codes=unknown,
            off_topic_subject=off_subject, used_context=used_ctx, parser="rules",
            demand_weight=weights.get("demand_weight"), capacity_weight=weights.get("capacity_weight"),
            operations_weight=weights.get("operations_weight"),
        )
