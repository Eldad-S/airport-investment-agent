"""Geographic groupings and colloquial airport aliases used to resolve user questions."""
from __future__ import annotations

STATE_NAMES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa", "KS": "kansas",
    "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland", "MA": "massachusetts",
    "MI": "michigan", "MN": "minnesota", "MS": "mississippi", "MO": "missouri", "MT": "montana",
    "NE": "nebraska", "NV": "nevada", "NH": "new hampshire", "NJ": "new jersey", "NM": "new mexico",
    "NY": "new york state", "NC": "north carolina", "ND": "north dakota", "OH": "ohio",
    "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania", "RI": "rhode island",
    "SC": "south carolina", "SD": "south dakota", "TN": "tennessee", "TX": "texas", "UT": "utah",
    "VT": "vermont", "VA": "virginia", "WA": "washington state", "WV": "west virginia",
    "WI": "wisconsin", "WY": "wyoming", "PR": "puerto rico", "DC": "district of columbia",
}

REGIONS: dict[str, tuple[str, ...]] = {
    "New England": ("CT", "ME", "MA", "NH", "RI", "VT"),
    "Mid-Atlantic": ("NY", "NJ", "PA", "DE", "MD", "DC", "VA"),
    "Southeast": ("NC", "SC", "GA", "FL", "AL", "MS", "TN", "KY", "WV", "AR", "LA"),
    "Midwest": ("OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"),
    "Southwest": ("TX", "OK", "NM", "AZ"),
    "Mountain West": ("CO", "UT", "NV", "ID", "MT", "WY"),
    "West Coast": ("CA", "OR", "WA"),
    "Alaska & Hawaii": ("AK", "HI"),
}

REGION_ALIASES = {
    "new england": "New England", "northeast": "New England", "n.e.": "New England",
    "mid atlantic": "Mid-Atlantic", "mid-atlantic": "Mid-Atlantic", "east coast": "Mid-Atlantic",
    "southeast": "Southeast", "south east": "Southeast", "the south": "Southeast", "dixie": "Southeast",
    "midwest": "Midwest", "mid west": "Midwest",
    "southwest": "Southwest", "south west": "Southwest",
    "mountain": "Mountain West", "rockies": "Mountain West",
    "west coast": "West Coast", "pacific": "West Coast", "left coast": "West Coast",
    "alaska and hawaii": "Alaska & Hawaii",
}

# Colloquial names -> IATA. Longest phrases are matched first.
AIRPORT_ALIASES = {
    "los angeles": "LAX", "l.a.": "LAX", "la": "LAX", "lax": "LAX",
    "john wayne": "SNA", "santa ana": "SNA", "orange county": "SNA",
    "san francisco": "SFO", "san fran": "SFO", "sf": "SFO", "frisco": "SFO",
    "anchorage": "ANC", "ted stevens": "ANC",
    "boston": "BOS", "logan": "BOS",
    "kennedy": "JFK", "laguardia": "LGA", "la guardia": "LGA", "newark": "EWR",
    "o'hare": "ORD", "ohare": "ORD", "o hare": "ORD", "midway": "MDW", "chicago": "ORD",
    "atlanta": "ATL", "hartsfield": "ATL", "denver": "DEN", "dallas": "DFW", "dfw": "DFW",
    "love field": "DAL", "seattle": "SEA", "seatac": "SEA", "sea-tac": "SEA",
    "vegas": "LAS", "las vegas": "LAS", "miami": "MIA", "orlando": "MCO", "phoenix": "PHX",
    "sky harbor": "PHX", "houston": "IAH", "hobby": "HOU", "nashville": "BNA", "austin": "AUS",
    "charlotte": "CLT", "philly": "PHL", "philadelphia": "PHL", "reagan": "DCA",
    "dulles": "IAD", "baltimore": "BWI", "detroit": "DTW", "minneapolis": "MSP", "twin cities": "MSP",
    "portland maine": "PWM", "portland me": "PWM", "portland": "PDX",
    "manchester": "MHT", "providence": "PVD", "hartford": "BDL", "bradley": "BDL",
    "burlington": "BTV", "bangor": "BGR", "worcester": "ORH",
    "honolulu": "HNL", "san diego": "SAN", "san jose": "SJC", "oakland": "OAK", "burbank": "BUR",
    "long beach": "LGB", "ontario": "ONT", "sacramento": "SMF", "salt lake": "SLC", "slc": "SLC",
    "tampa": "TPA", "fort lauderdale": "FLL", "ft lauderdale": "FLL", "new orleans": "MSY",
    "st louis": "STL", "saint louis": "STL", "kansas city": "MCI", "pittsburgh": "PIT",
    "cleveland": "CLE", "cincinnati": "CVG", "columbus": "CMH", "indianapolis": "IND", "indy": "IND",
    "raleigh": "RDU", "raleigh durham": "RDU", "san antonio": "SAT", "albuquerque": "ABQ",
    "tucson": "TUS", "el paso": "ELP", "boise": "BOI", "reno": "RNO", "milwaukee": "MKE",
    "jacksonville": "JAX", "memphis": "MEM", "louisville": "SDF", "richmond": "RIC",
    "buffalo": "BUF", "albany": "ALB", "syracuse": "SYR", "rochester": "ROC", "omaha": "OMA",
    "palm springs": "PSP", "kahului": "OGG", "maui": "OGG", "kona": "KOA", "fairbanks": "FAI",
    "juneau": "JNU", "san juan": "SJU", "savannah": "SAV", "charleston": "CHS", "myrtle beach": "MYR",
}

# Multi-airport groupings people use colloquially.
METRO_GROUPS = {
    "nyc": ("JFK", "LGA", "EWR"), "new york": ("JFK", "LGA", "EWR"),
    "bay area": ("SFO", "OAK", "SJC"), "socal": ("LAX", "SNA", "BUR", "LGB", "ONT", "SAN"),
    "dc": ("DCA", "IAD", "BWI"), "washington dc": ("DCA", "IAD", "BWI"),
}

# Common English / chat words that are also airport codes: only match these when typed in
# UPPER CASE (e.g. 'SUN', 'DAY', 'LAW' are real codes).
AMBIGUOUS_LOWERCASE = set("""
act add age ago aid aim air all and any app are arm art ask ate bad bag bar bay bed bet big bit
box boy bus but buy bye can car cat cut day did die dig dog dry due eat end era etc eye far fat
few fit fly for fun gap gas get god got gun guy had has hat her hey him his hit hot how ice ill
its job joy key kid lab lag law lay led leg let lie lol lot low mad man map max may men met mid
min mix mom net new nor not now odd off oil old one our out own pay per pet pop pro put ran raw
red rid row run sad saw say see set she sit six sky son sum sun tax tea ten the tie tip too top
toy try two use via war was way wet who why win won yes yet you zip pls plz thx omg wtf idk imo
tbh btw lmk fyi wat hmm yep nah ok wow hub den jan dec sat elm cod pie mob ash lit
""".split())
