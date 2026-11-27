# Week 4 Devnotes: Starting the Model

Working notes. This is the list of what was done, kept up to date as the week
runs; the write-up follows once the work is finished.

## Scope

- Week 4 starts from the week-3 state: eight matchers, the harness, the
  inspector, and the reference sets — with no model of our own yet.
- Decision: week 4 is about creating a new model, not about improving the
  existing matchers or extending the diagnosis tooling.

## Model definition

- The model takes a pair of images — one query, one reference — and outputs
  keypoints on both plus the correspondences between them.
- The difference from existing matchers: keypoints must sit on the **outline of
  the object**, avoiding low-texture and non-descriptive regions.
- Training budget is deliberately undecided. The idea is validated by
  measurement first; training only happens if the measurement supports it.

## Design

- Wrote [`docs/model-design.md`](../model-design.md): problem statement, a
  falsifiable claim, the input/output contract, why boundary points should help,
  where the boundary signal comes from, three architecture options, training data
  requirements, Phase 0 measurements, evaluation, milestones, and risks.
- Claim under test: boundary-anchored keypoints give an object-consistent
  correspondence set and a better-conditioned geometry than
  repeatability-driven keypoints.
- Falsification: if restricting existing detectors to object regions does not
  improve the hard pairs, or boundary points are too unstable to match, the idea
  does not hold for this data.
- Motivating evidence recorded from the week-3 results:
  - 24% of the remaining hard pairs have predicted boxes 20× too large and
    centred a box-width away — background matches winning RANSAC.
  - 61% of the hard pairs are recoverable by correcting only scale and
    translation, the signature of an under-constrained geometry.
- Architecture options, in order of cost: (A) boundary-aware detector inside the
  existing pipeline, (B) joint boundary-supervised detector and descriptor,
  (C) end-to-end pair model. The plan assumes A first.
- Added metrics specific to the claim: outline precision and object-anchored
  match rate, reported alongside hit rate so a gain cannot come from something
  other than the stated idea.

## Model scaffold

- Adopted the `src/` layout and packaged the project as `osid`, so the model
  sits next to the benchmark it will be measured against.
- Built a per-image detector and descriptor: frozen DINOv3 ViT-S/16 backbone
  tapped at four layers, a fusion decoder to stride 4, a boundary head and a
  descriptor head, and local-maxima keypoint selection. Matching is a separate
  stage (`adapter.py`) so detection can be measured and cached on its own.
- 493k trainable parameters; the backbone stays frozen and in eval mode.
- Documented the architecture with a diagram and a stage → file table.
- Built a synthetic data generator of controlled difficulty: one filled shape on
  a flat background, with per-view size, rotation, perspective, contrast, blur,
  noise, and an optional shared background. Five shape families (polygon,
  rounded polygon, smooth blob, star, ellipse) over an 8-50% radius range, so
  the model sees curved and straight boundaries at many scales.
- Masks and correspondences are exact by construction: vertex reprojection under
  the homography is 1e-13 px, and mask-alignment IoU averages 0.980 (min 0.919,
  edge rasterisation on the smallest shapes).
- Added a contrast x blur grid so difficulty can be inspected directly rather
  than inferred from the parameters.

## Phase 0 (not started)

- Pseudo-masks from the reference boxes, then measure where current keypoints
  land: on the instance, on its boundary, or in the background.
- Repeat for the matches RANSAC accepts versus rejects.
- Oracle region filtering on the 251 hard pairs, with the GT box on the query as
  the upper bound.
- Boundary-band filtering, which is the model's actual operating point.

## Open items

- Run Phase 0 measurement 1 and decide whether the premise survives.
- Decide whether pseudo-masks come from GrabCut (no new dependency) or a
  promptable segmentation model.
- Decide the training scope after Phase 0: instance-level only, or instance plus
  category.
