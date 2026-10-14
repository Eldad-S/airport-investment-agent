# Data sources

All sources below were **actually downloaded and processed** on 2026-10-14 by `scripts/build_dataset.py`.
Retrieval timestamps, URLs, row counts and QA statistics are recorded in `data/processed/manifest.json`.
All data is **historical**, not real-time.

## 1. BTS Reporting Carrier On-Time Performance

| | |
|---|---|
| Provider | U.S. DOT, Bureau of Transportation Statistics (TranStats) |
| Retrieval | `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YYYY}_{M}.zip` - public monthly zips, no authentication (~25-33 MB each) |
| Coverage used | Jan 2024 - Dec 2025 (24 months). Data was available through Jul 2026 at retrieval; calendar years were chosen to align with FAA data. |
| Volume | 14,080,680 flight records; 358 airports; 0 duplicate flight keys; 0 invalid airport codes |
| License | U.S. government public data |
| Processed outputs | `ontime_airport_month.parquet` (8,112 rows), `ontime_route_month.parquet` (139,287 rows) |

**Fields used:** `FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest, CRSDepTime,
CRSArrTime, DepDelayMinutes, DepDel15, TaxiOut, TaxiIn, ArrDelayMinutes, ArrDel15, Cancelled, Diverted,
Distance, NASDelay`.

**Processing:** identifiers normalized (upper-case, ICAO `Kxxx` -> IATA), rows with invalid codes dropped and
counted, de-duplication on (date, carrier, flight number, origin, destination, scheduled departure). Each flight
is reported once by the operating carrier, so codeshare marketing duplicates do not appear. Departure metrics
are attributed to the origin, arrival metrics and NAS delay to the destination (BTS convention). Peak-hour
movements = busiest scheduled hour of each day (deps + arrs), averaged over days.

**Known gaps:** domestic only; only carriers with >= 0.5 % of domestic scheduled passenger revenue (excludes
many regional-only, all foreign, charter and cargo carriers). Delay causes are reported only for flights >= 15
minutes late.

## 2. FAA Passenger Boarding (Enplanement) Data

| | |
|---|---|
| Provider | Federal Aviation Administration, Air Carrier Activity Information System (ACAIS) |
| Retrieval | `https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/arp-cy2025-commercial-service-enplanements.xlsx` (CY2025 + CY2024 columns) and `.../cy19-commercial-service-enplanements.xlsx` (CY2019 baseline) - public, no authentication |
| Coverage | Commercial-service airports (>= 2,500 annual enplanements): 519 rows |
| License | U.S. government public data |
| Processed output | `faa_enplanements.csv` |

**Fields:** `Locid` (FAA LID), `Hub` (L/M/S/N), `S/L` (service level), `CY 25 / CY 24 / CY 19 Enplanements`.

**Identifier crosswalk:** FAA LIDs equal the IATA code for 479 airports; 36 were mapped through OurAirports
`local_code` (e.g. `IWA` -> `AZA` Mesa Gateway, `GPI` -> `FCA` Glacier Park); 4 rows stay unmapped (3 blank /
footer rows and Kake, AK, which has no IATA code) and are kept but not joined.

**Known gaps:** annual totals only; domestic and international passengers combined; no cargo.

## 3. OurAirports

| | |
|---|---|
| Provider | OurAirports (community-maintained) |
| Retrieval | `https://davidmegginson.github.io/ourairports-data/airports.csv` |
| License | Public domain |
| Processed output | `airports_reference.csv` (airports referenced by the other sources) |

**Fields:** `iata_code, local_code, name, municipality, iso_region (-> state), latitude_deg, longitude_deg, type`.
Used for names, states / regions and great-circle distances. BTS's own `Distance` field is used only as a
cross-check (max difference for ANC routes: 10 mi).

## 4. Curated context notes (`data/context/airport_notes.json`)

Five short, cited qualitative notes (SFO, SNA, LAX, ANC, BOS) used only as *hypotheses* in explanations, never
in any calculation. Each note links to the public source to verify.

## Sources investigated but not implemented

| Source | Why not (yet) |
|---|---|
| BTS T-100 Segment (seats, load factor, international) | Download requires TranStats form interaction; first addition with more time - enables load-factor evidence for unmet demand |
| BTS DB1B (fares) | Large; needed for fare-premium evidence of constrained supply |
| FAA ASPM / OPSNET | ASPM requires an FAA account; would provide airport acceptance rates and ground-delay programs |
| OpenSky Network | Real-time / state-vector data; rate-limited and not needed for annual investment screening |

## Fallback mode

The app reads the committed snapshot only. If a refresh fails, `build_dataset.py` stops with a
`DataSourceError` naming the failing URL and the existing snapshot is left untouched. No synthetic data is ever
substituted. Test fixtures in `tests/conftest.py` are synthetic and clearly labeled as such; they are never
used by the app.
