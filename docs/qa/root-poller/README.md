# Root poller presentation QA

[2026-11-13 capture manifest](2026-11-13/manifest.json) binds actual root summary,
browser and typed/legacy detail captures to their input/source hashes. There are
12 view/width combinations, each with text, HTML and SVG. The requested 80/120/160
column widths and SVG geometry are checked; these are not three independent
scientific evaluations.

Selected summary (80 columns), typed detail (120) and legacy detail (120) SVGs were
rasterized locally and visually inspected for readable content and structure.
Unicode border alignment can vary with local SVG fonts; geometry checks are not a
claim of identical pixels on every platform. The Rich/plain text is independently
captured. See the [root-helper guide](../../engineering/root-poller-presentation.md).

Inputs are synthetic computational/legacy controls, not real observations. This is
a post-beta2 follow-up snapshot; historical release/capture artifacts are unchanged.
Full-menu QA, source restoration and scientific validation remain separate work.
