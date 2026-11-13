# Menu QA evidence and limits

Each capture bundle records its own code/input scope, synthetic or live basis,
widths and source/artifact hashes. It is not a certificate for every menu feature,
source availability or scientific validity. Plain text, HTML and SVG are separate
outputs; checking one does not prove the others render correctly.

## Correction: historical coloured capture widths (#69)

The shared capture helper originally supplied width but not height to Rich. Under
`TERM=dumb`, the installed Rich version clamped forced-terminal coloured output to
80 columns, including when 120 or 160 had been requested. Plain-text captures still
used their requested widths. Therefore earlier bundles using that helper do **not**
establish three distinct coloured SVG/HTML width layouts in that environment. Their
hashes describe saved bytes, not verified console geometry.

Keep those historical artifacts intact. Their source snapshots and plain-text checks
remain evidence within their stated scope; do not silently reinterpret them as newly
verified wide graphical layouts. #69 tracks the remaining inventory/re-capture work.

The corrected helper specifies both dimensions and rejects an unhonored width.
A `TERM=dumb` regression checks actual SVG geometry at 80/120/160. The new
`source-freshness/2.2.0.dev0` bundle uses this helper and separately checks terminal
cell widths. Other historical bundles have not yet been re-captured or visually
reviewed with the correction. This limitation does not demonstrate a defect in every
production menu; it is a confirmed defect in the prior capture evidence.

A read-only inventory at intermediate head `4f3f258` measured **84 labelled SVGs**:
40 matched their advertised default Rich geometry and 44 did not. Nine matches
belong to the new receipt bundle; older bundles are 31/75 matches and 44/75
mismatches. Initial menu-entry graphics match; do not call all older artifacts
broken. The per-bundle matrix and hash-linked inventory are recorded in #69.
Geometry is not a substitute for content, navigation or scientific-data review.
