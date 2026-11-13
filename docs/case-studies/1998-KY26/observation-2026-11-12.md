## New dated provider observation — not a historical-baseline replacement

One sequential SBDB request on 2026-11-12 16:43:10 UTC used `sstr=1998 KY26&full-prec=true&phys-par=true&cov=mat`, an application-specific identifying User-Agent and no undocumented nongrav parameter. The provider returned schema **1.3**, SPK-ID **50012415**, orbit solution **36**, solution date **2024-05-07 09:53:57**, observation arc **1998-05-31–2024-05-06**.

| Fitted parameter | Native value (au/d²) | Native sigma (au/d²) | Kind |
|---|---:|---:|---|
| A1 | 3.655395414270661E-11 | 3.867E-11 | EST |
| A2 | -2.196384257496804E-13 | 4.866E-14 | EST |
| A3 | 1.686333710048786E-12 | 1.121E-12 | EST |

The response also contains fixed model constants (ALN, NK, NM, NN, R0), which must remain distinct from fitted accelerations.

**Important covariance contract:** displayed element epoch is JD(TDB) **2461200.5**, whereas the returned 9×9 covariance epoch is **2457098.5**. Labels are `[e,q,tp,node,peri,i,A1,A2,A3]`. Do not treat it as a covariance of `[a,e,i,...]` at the displayed epoch. Use its labelled solution-epoch elements or explicitly propagate to a common epoch. Cross-component correlations must survive, not become independent per-coefficient tests.

The current-retrieval physical parameters include H=25.91±0.03, diameter=0.011±0.002 km, rotation=0.089193±0.000002 h, albedo=0.52±0.08, citing Santana-Ros et al. (2025) in the response. This is a catalogue observation, not full loading of the paper/alternative fits.

Raw response captured locally as an immutable new observation; SHA-256 `59d397a914ab9c2882f8106b1530fbad572308d5eedf1d54f5ce120e09404993`. Raw redistribution review is pending, so this comment publishes the factual extracted metadata, not the full payload. Original baseline remains unchanged.

Running these newly retrieved full-precision elements through the archived detector gives legacy score **4.142377694114837**, uncalibrated posterior **0.03674337652291626**, heuristic combined p **0.0012968347752451148**. This reproduces the displayed 4.142 rounding but not the exact reported historical combined p; it does not recover the original retrieval provenance. No tuning was performed.

Sources: [SBDB request](https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=1998%20KY26&full-prec=true&phys-par=true&cov=mat), [provider contract](https://ssd-api.jpl.nasa.gov/doc/sbdb.html).

These are fitted parameters, not proof of thrust or origin. Historical 2023 A3 estimates and the 2026 preprint observation set require separate source/epoch/model comparison before any agreement or contradiction is asserted.
