# NEODyS evidence: acquisition, photometry and honest limits

Application **2.2.0.dev0**, unreleased draft stack. This increment addresses #63
and part of #40/#16; it does not complete the scientific investigation.

## What was wrong

The previous adapter tried to turn a provisional designation into a permanent
number through an MPC *name* lookup. That dependency could fail even when NEODyS
served the object's provisional `.eq0` file. The parser emitted H/G compatibility
values but the normalizer omitted their typed physical evidence. It did not bind
the native object identifier to the requested file, and repeated orbital records
could become last-record-wins. The separate legacy HTTP helper also manufactured
zeroes for absent elements from an unverified REST response.

## What now happens

`DataFetcher.fetch_neo_data()` uses the existing acquisition/reconciliation/local
ledger, not another engine or endpoint. NEODyS accepts a validated permanent number
or compact unpacked provisional identifier directly. Names and number-plus-name
aliases use the bounded, native-identifier-checked MPC resolver; ambiguous results
cannot choose a file. This does not create a global canonical alias registry.

The accepted file must contain one matching native object, an OEF2.0 multi-line
header in ECLM/J2000, one supported elliptic EQU state and one labelled MJD epoch.
Unsupported frames/layouts or conflicting states fail closed with the raw packet
retained. Explicit 404, malformed data, ambiguity and transport/service failure
remain different outcomes. Requests have a configured timeout, application header,
no redirect-following and no hidden retry; this is not a provider-wide pacing or
end-to-end deadline guarantee. Responses are closed. The parser admits at most
2 MB of decoded text; HTTP acquisition itself is buffered, not a streaming byte cap.
Directory health checks remain directory reachability, not proof an object query
or all source features work.

The existing helper names remain available. `_make_request()` is a controlled
legacy projection through validated OEF acquisition; it no longer queries the
undocumented nominal-REST path or invents zero elements. `neodys_rest_url` remains
a legacy setting, not a claim that this new path uses it. `_resolve_number()`
retains a permanent-number purpose; ordinary provisional acquisition does not
require it. Epoch compatibility tests now use a correctly identified/headered
synthetic file; MPC resolver tests assert the exact bounded lookup and positive
native identity, rather than merely asserting a result differs from the year.

## Photometry and provenance

A native `MAG` line contains H and G. H is normalized into the existing
`absolute_magnitude_h` field, with the original line, source, file hash, reference
frame and orbit-epoch context retained in an atomic assertion. A NEODyS record is
not disguised as an SBDB `phys_par` response. Existing SBDB assertions and their
replay format remain unchanged.

Missing/invalid H or an uninterpretable H/G pair does not become zero. A real zero
is retained. Distinct MAG records remain alternatives; duplicate records are
labelled duplicates, not independent observations. Ambiguity produces no selected
H, not an average or the last row. No uncertainty, measurement date or fitted-versus-
fixed status is invented. G stays in the native record: the existing physical
model has no typed phase-slope field or phase-law fit. H and G are not independent
origin evidence. An orbit epoch is not the date of a photometry measurement.

The parser conservatively recognizes OrbFit's `-9.99` undefined-magnitude marker,
without treating other negative H values as missing. This marker is documented in
the public OrbFit source inspected below; the older source is not proof of every
current provider convention. Other unsupported layouts remain explicit, and no
complete sentinel catalogue is claimed.

Every non-comment data record is accounted for. EQU/MJD are normalized; MAG records
report whether H was selected; COV/COR/NOR/LSP stay native with a reason; unknown
records stay unsupported/native. This is **not** a covariance transformation or
NG fit. An equinoctial covariance must not be relabelled as a Keplerian matrix,
and LSP must not silently become zero A1/A2/A3 measurements. Those data still need
an appropriate typed basis, ordering, units, Jacobian and validation (#43/#5).

## Reading results and preserving history

Object review, plain text and Markdown show the NEODyS source label, H, unavailable
uncertainty, native H/G record and unresolved alternatives. Empty photometry gets
an explicit unavailable message rather than a blank section. Existing graphical
ranking decomposition remains a heuristic contribution chart, not a probability.

New acquisition cache contexts include this normalization-contract revision so old
cached omissions do not mask the repair. Old query/history records are not rewritten;
replay remains tied to their frozen adapter packets. Cache eviction is still not a
backup. The repository archives assertions/row values, not original database bytes.
The exact requested key remains distinct from a verified cross-query identity.

## Reproduce

```sh
python -m pytest tests/test_neodys_evidence.py -q
python scripts/qa_neodys_evidence.py --output /tmp/aneos-neodys-new-capture
```

The capture directory must not already exist. It runs the actual public query,
disposable file-SQLite export and object review with **synthetic** available,
ambiguous and missing-photometry OEF packets. It produces text/HTML/SVG at
80/120/160 columns, JSON/Markdown reviews, full evidence exports and hashes.
Captured artifacts are in `docs/qa/neodys-evidence/2.2.0.dev0/`.
Analytic-state reconstruction tests cover circular/elliptic and low/high-inclination
coordinate cases. They validate normalization, not orbit determination or a fit.

## Primary references and limits

- [Astroquery's NEODyS guide](https://docs.astropy.org/projects/astroquery/en/stable/solarsystem/neodys/neodys.html)
  demonstrates provisional identifier files and distinguishes orbital state,
  covariance and photometry.
- [Astroquery's NEODyS API](https://astroquery.readthedocs.io/en/latest/api/astroquery.solarsystem.neodys.NEODySClass.html)
  identifies MAG as the H/G pair. The installed implementation was also inspected;
  the documentation's epoch terminology does not override the file's literal MJD
  and declared time scale.
- [OrbFit distribution and scope](https://adams.dm.unipi.it/~orbmaint/orbfit/)
  warns that orbit/covariance formats can include additional dynamical parameters.
  Do not assume every covariance is 6×6.
- [Public OrbFit orbit-elements source](https://adams.dm.unipi.it/~orbmaint/orbfit/OrbFit/src/propag/orbit_elements.f90)
  contains the undefined magnitude/default phase-slope initialization and format
  routines. The directory dates this copy to 2017; current full-archive inspection
  was not completed (a bounded download exceeded 20 MB). This is an explicit scope
  limit, not certification of the provider's current internals.

Raw live source files are kept outside Git pending redistribution review. Provider
availability, scientific measurement quality, compatible-epoch comparisons,
independent calibration and every-menu readiness remain separate requirements.
Nothing here establishes artificial origin, a calibrated probability or five sigma.

## Delivery verification

Final local offline suite: **1,495 passed**, eight network tests excluded; core
coverage **50.21%**, above the unchanged 30% floor. The new source/normalizer gate
passes **91 tests at 97.37%** against 90%. The existing physical-evidence gate,
expanded with these source cases, passes **109 tests at 91.81%**, retaining 90%.
The new regression file has 73 cases, including atomic cache/SQL/export replay,
actual renders, ambiguity/outage/missing controls, analytic coordinate reconstruction
and a prior-commit synthetic SBDB assertion fixture that replays unchanged.
Source/artifact hashes and all nine terminal-width bounds were verified. These
are bounded engineering results; the published revision's CI must also pass.
