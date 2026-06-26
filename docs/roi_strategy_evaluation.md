# Unified ROI Strategy Evaluation (57 single tracks)

> **Evaluation phase only.** This step measures where the melt pool sits and
> compares ROI / analysis-window options. It does **not** crop or write any ROI
> `.npy`, does not modify the raw matrices, and runs no feature extraction / ML /
> response-surface work. The formal ROI is generated only after the user reviews
> the QC figures and confirms a strategy.

Pipeline: `scripts/03a_evaluate_roi_strategy.py`
(core algorithms in `src/processing/hot_region_mask.py` and
`src/roi/roi_evaluation.py`; QC figures in
`src/visualization/roi_qc_figures.py`).

## Inputs & conventions

- Source matrices: `data/processed/matrix/<sample_id>.npy` (float32 °C, N×512×640),
  read-only (mmap). `dataset.npy` is excluded.
- **Effective frames = `frames[1:]`** (frame 0 is the camera startup frame).
- Spatial scale 0.0332889481 mm/px; valid band 300–1800 °C; hard saturation
  ≥6500 °C (from `configs/physical_calibration.yaml`).
- Image geometry: scan axis = Y, melt pool moves toward image **up** over time
  (array row index decreasing). All bboxes/rects are half-open
  `(top, left, bottom_excl, right_excl)`.

## Temperature spatial-localization policy (geometry only)

The raw matrix is never modified. Per frame, the 700 °C envelope and 800 °C core
main regions are localized as follows:

1. threshold mask `T >= level` (numerically includes above-range / hard-sat);
2. 8-connected components; components `< min_component_area` (9 px) dropped;
3. a component is **genuine** iff it contains ≥1 in-range *valid* pixel ≥ level;
   pure above-range / hard-saturation components are **isolated splatter** and
   excluded (counted as `above_range_isolated`);
4. the largest genuine component is the **main body**, hole-filled so an embedded
   hard-saturation core never makes a hole (mask only);
5. above-range pixels continuous with the main body are **retained** for geometry
   (`above_range_connected`); 2739/3085/6553.5 °C are never read as real
   temperatures.

## Results (57 tracks; see `results/qc/roi/roi_strategy_summary.json`)

- **Legacy ROI** `top=200,left=0,h=300,w=600` → rows [200,500), cols [0,600):
  min 800-core coverage **100%**, min 700-envelope coverage **99.71%**
  (fails the ≥99.9% bar). `R2_T3` clips the envelope top and has a negative edge
  distance (the pool reaches row 195). **Legacy ROI is NOT recommended.**
- **Raw global 700 bbox** over all 57 tracks: `(195, 106, 473, 312)`.
- **Candidate fixed global ROI** (700 union + 20 px margin, rounded to ×8,
  clamped to 512×640): `top=175, left=86, bottom=495, right=334` →
  **320×248 px (10.65×8.26 mm, 24.2% of the frame)**, min 700/800 coverage
  **100% / 100%**, all safety margins ≥10 px (no edge-fail samples), no track
  touches a frame edge.
- **Melt-pool vertical travel** ≈ 38–53 px (~1.3–1.8 mm); per-frame 700 bbox
  ≈ 173×145 px (p95) / 181×169 px (p99).
## Fixed tracking window — corrected sizing (extent-based)

The window is the **same fixed size for every track and frame**; only its centre
moves with the melt-pool centroid (no scaling, no rotation). Near a frame edge it
is translated to stay inside the image (`edge_adjusted`), never resized.

### Old (percentile-of-bbox-size) window — REJECTED

The first design sized the window from the p99 of the per-frame 700 bbox
**width/height** plus a margin, then centred it on the temperature **centroid**.
Because the centroid is *not* the bbox geometric centre, an asymmetric main
region extends further on one side than the window's half-size:

- old window: **192×208 px**
- old clipped frames: **35** (700 envelope) and **1** (800 core) — the main
  region is clipped, so this window was **not** acceptable for features.

### New (four-direction extent) window — ACCEPTED

The corrected algorithm measures, per effective frame, the four placement-aware
extents from the (float) centroid to the cleaned main-body bbox edges
(`round(cx) − left`, `right_excl − round(cx)`, and the vertical pair), takes the
worst case over all frames, doubles it and rounds up to a multiple of 8:

- `left/right/top/bottom_extent` stats (min / p95 / p99 / max) are recorded in
  the JSON (`centroid_to_bbox_extent_stats_px`);
- new window: **256×216 px (8.52×7.19 mm)**;
- **700 envelope coverage 100%, 800 core coverage 100%, clipped_frame_count = 0**,
  edge-adjusted frames = 0;
- window area (55,296 px) is **70%** of the candidate global ROI (79,360 px) — it
  still isolates local melt-pool morphology, so it is not "too large".

> Acceptance rule: a tracking window may be recommended for formal features only
> when it covers the cleaned main 700 **and** 800 region of every frame with
> `clipped_frame_count = 0`. If the smallest 100%-coverage window approaches the
> global ROI (≥85% of its area) or exceeds the frame, the strategy falls back to
> `single_global_fixed_roi` with `tracking_window_status = auxiliary_only`.

## Recommendation

**`global_roi_plus_tracking_window`** (two-layer); `tracking_window_status =
candidate_full_coverage`:

- a single fixed global ROI covers 100% but is ~78% per-frame background (the
  pool fills only ~22% of it and sweeps ~45 px upward), so
- use the **fixed global ROI** for absolute position / trajectory / signed
  centre offset / global stability, and the **extent-based 256×216 px tracking
  window** (100% 700+800 coverage, 0 clipped) centred on the melt-pool centroid
  for per-frame morphology / width / length / local temperature distribution /
  scan-direction & transverse gradients / asymmetry.

Option A (single fixed global ROI) remains valid and 100%-covering if only
position/coverage metrics are needed; option C (full 512×640) is unnecessary
here (no edge-touch, no manual-review samples). Original centroid positions are
always preserved separately — the tracking window must not hide real position
offset.

## Machine-readable record

The accepted evaluation result is recorded in `configs/roi_strategy.yaml` as the
formal ROI-strategy configuration. This config is the machine-readable authority
for the evaluated parameters, but its activation status is
`evaluated_not_activated`; formal ROI generation and formal feature extraction
remain disabled. The local JSON under `results/qc/roi/` is provenance only and
is not required for configuration loading.

The earlier **192×208 px** tracking window remains a legacy/rejected candidate,
not approved for formal processing.

## Outputs (LOCAL only, not committed)

- `results/tables/roi_bbox_by_track.csv` (57 rows)
- `results/tables/roi_exception_list.csv`
- `results/tables/roi_repeatability_summary.csv` (19 rows)
- `results/tables/tracking_window_coverage_summary.csv`
- `results/qc/roi/roi_strategy_summary.json`
- `results/qc/roi/*.png` (+ `.pdf` for overlays / comparison) — 12 QC figures.

## Not done in this phase

No formal ROI crop, no ROI `.npy`, no temperature-field feature extraction
(script 10), no scripts 13–16, no ML, no response-surface fitting, no
cooling-rate / AUC, no modification of raw matrices.
