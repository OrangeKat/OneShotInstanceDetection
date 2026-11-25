# Devlogs

One devlog per week. Each entry covers what was built, what failed, what was
learned, and what comes next. The same notes are summarized in the weekly
LinkedIn post.

| Week | Topics | Link |
|---|---|---|
| Week 1 | goal, design, datasets, metrics, SIFT/ORB/AKAZE baselines, FlickrLogos-32 failure analysis | [week1-devnotes.md](week1-devnotes.md) |
| Week 2 | HPatches validation in the bench, MMA/HEA/AUC/corner error, the eight-matcher suite (SIFT/ORB/AKAZE, SuperPoint/ALIKED + LightGlue, LoFTR, RoMa v1, FlexGlue), expected-vs-actual gap, 480 px protocol audit and re-runs, RoMa balanced-sampling fix, per-sample diagnostics and stage/slice analysis, PRW substitution for CUHK-SYSU, final Flickr/RPC/HPatches/PRW results, PRW complexity evidence, inline metric definitions, worked example | [week2-devnotes.md](week2-devnotes.md) |
| Week 3 | diagnosis scope reset, model agreement and hard pairs (Flickr 381/2208, RPC 307/886), reference-quality audit (six defective classes, 48% of hard pairs), reference replacement rerun (hard pairs 381 → 251, every model +15–38% hit rate), pair inspector with pre-RANSAC correspondences, residuals, keypoints, Grad-CAM and occlusion, reference picker with `check-refs`, ablation queue, browser smoke test | [week3-devnotes.md](week3-devnotes.md) |
