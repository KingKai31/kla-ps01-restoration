# Stage B risk checks — hallucination and SAR domain-gap

Two specific risks flagged before the PPT: (1) DIV2K/Flickr2K training data
could bias the model toward natural-photo texture priors, the same
hallucination-risk class that ruled out Real-ESRGAN, just introduced via
training data instead of loss function; (2) SAR (Sentinel-1&2 terrain
imagery) was justified by shared noise physics, but its *content* domain
(terrain) is structurally very different from KLA's semiconductor
dendrites/textures. Both checked with real evidence below, not assumed.

## 1. Natural-photo hallucination check

**Method:** pulled the worst-performing Stage B val/OOD-proxy samples —
bottom-5 by SSIM ∪ top-5 by LPIPS (10 unique images, since these are
measurably different failure populations: the worst-SSIM group actually has
*low* LPIPS 0.17–0.24, the worst-LPIPS group has much higher LPIPS
0.36–0.51 with less catastrophic SSIM) — and visually inspected each for
fabricated organic texture or photographic-looking artifacts not present in
GT, as distinct from the already-known texture-oversmoothing failure mode.
Full grid: `reports/figures/stageB_worst_case_hallucination_check.png`.
Per-image numbers: `reports/stageB_val_per_image_metrics.csv`.

**Finding: no evidence of natural-photo hallucination in any of the 10
cases. The existing oversmoothing explanation accounts for all of them.**

- Worst-SSIM group (000905, 003070, 003069, 003071, 002607): all GT images
  are fine-grained stochastic/granular texture (the same failure mode
  documented for Stage A). Output is a smoothed version that tracks GT's
  coarse structure but loses fine detail — it does not invent new content,
  it under-represents real content.
- 001781 (worn/scratched surface): output preserves the actual scratch/crack
  line from GT, smooths the fine granular surface texture around it. Same
  pattern.
- 002626 (leaf/frond) and 001142 (wood-grain/fabric stripes): output tracks
  GT's real structure (leaf silhouette and veins, stripe pattern) with some
  detail loss. No invented structure.
- **002208 (brick wall) — the single worst LPIPS score of the whole set
  (0.510), and the most natural-photo-like image among all 10.** This is
  the case most worth scrutinizing for the hallucination concern
  specifically. Visual inspection: the output's brick/mortar layout closely
  matches GT's actual brick pattern — same edges, same course lines, same
  general grout texture. It is not inventing a plausible-but-wrong brick
  pattern; it's reconstructing the real one with somewhat cleaner, less
  rough surface detail than GT. The high LPIPS here looks like it's
  penalizing loss of fine surface roughness/perceptual texture, not
  fabricated content.
- 001002 (cloud/mottled sky): output is a smoothed version tracking GT's
  general cloud shape. Same oversmoothing pattern.

**Read for the slide:** the hallucination risk was real to check, and
checking it mattered — but on this evidence it hasn't materialized. The
one image where it would most plausibly show up (a real photograph, worst
LPIPS in the set) instead shows accurate reconstruction with detail loss,
not fabrication. State this as checked-and-not-observed, not as "ruled
out" — 10 images is a meaningful spot-check, not exhaustive coverage.

## 2. SAR domain-gap check

**Method:** no source-category labels exist in the KLA data (only the
unsupervised-clustering OOD-proxy split), so a clean SAR-vs-non-SAR
breakdown isn't directly available. Checked whether the two held-out val
clusters (2 and 9) are internally homogeneous enough to serve as a proxy —
they aren't: both are visually mixed (cluster 2: cobblestone pavement,
leaves, sky, architecture, ships; cluster 9: statue/rock texture, cloud,
skyline gradients — see `reports/figures/cluster_2_val_preview_32.png` and
`cluster_9_val_preview.png`). Instead, manually identified the specific
images within cluster 2 that are genuinely structurally analogous to SAR's
repetitive overhead terrain patterns (cobblestone pavement, plaza ground,
mottled/ridged ground textures) — **10 images**: `000017, 000018, 000019,
000020, 000032, 000033, 000034, 000035, 000178, 000401`. Compared their
Stage B val metrics against the rest of val (n=496).

| Group | n | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Terrain-like subset | 10 | 27.25 | 0.7459 | 0.1508 |
| Rest of val | 496 | 28.10 | 0.7309 | 0.1630 |
| Full val | 506 | 28.09 | 0.7312 | 0.1627 |

**Finding: the terrain-like subset does not underperform the rest of
val — SSIM and LPIPS are both marginally *better* than average, PSNR is
comparably close (individual values span 23–32 dB, so the ~0.9dB gap is
within normal per-image spread, not a systematic drop).** This is
reassuring, but **n=10 is a small, hand-curated sample** — real confidence
would need either KLA's actual source labels or a dedicated SAR-only
validation slice, neither of which we have.

**Read for the slide:** this rough check is evidence, not proof, and should
be framed that way. Combined with the small sample size, the honest
position is still to reframe the SAR justification precisely rather than
oversell it:

> SAR (Sentinel-1&2) was included in Stage B's training mix specifically
> for its **shared multiplicative speckle noise physics** with the target
> semiconductor imagery — not for content similarity. Its actual content
> (terrain: agricultural, barren, grassland, urban) is structurally
> different from semiconductor dendrites/textures, and we treat that as a
> real, acknowledged gap, not a seamless match. The hypothesis was that
> shared noise statistics would transfer despite the content difference; a
> small rough check (10 terrain-like images in the validation set) showed
> no performance degradation on that subset, which is consistent with the
> hypothesis but not a rigorous confirmation given the sample size.

This is a testable, honest claim — precise about what was hypothesized,
what was checked, and what remains unverified — which holds up better
under judge scrutiny than implying SAR was an obviously strong content
match.
