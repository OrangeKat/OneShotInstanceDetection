# Model design: boundary-anchored keypoint matching

Status: design draft, week 4. No implementation yet — Phase 0 is a measurement
that decides whether the idea holds.

## 1. Problem

Given a reference crop of one instance and a query image, predict keypoints on
both images and the correspondences between them, where the keypoints lie on the
**outline of the object** rather than anywhere the detector finds local contrast.

Standard detectors optimise repeatability: corners, blobs, and texture anywhere
in the image. On a reference crop that includes background, and on queries full
of clutter, most of the keypoint budget goes to surfaces that are not the object.
Two consequences show up in the benchmark:

- Matches on background dominate the consensus set, so RANSAC recovers a plane
  that is not the object. In the current results, 24% of the remaining hard pairs
  have predicted boxes 20× too large and centred a box-width away — the
  homography locked onto the background.
- With few object-anchored matches, the geometry is under-constrained. On the
  same hard set, 61% of the failures are recoverable by correcting only scale
  and translation, which is what an under-constrained similarity estimate looks
  like.

The hypothesis: keypoints constrained to the instance outline give a
correspondence set that is object-consistent by construction, which improves
both the match and the geometry it implies.

## 2. Claim and falsification

**Claim.** A detector that places keypoints on object boundaries, and refuses
flat or non-instance regions, produces matches that are more often correct and
a homography that is better conditioned than the same matcher fed with
repeatability-driven keypoints.

**Falsified if.** In Phase 0, restricting existing detectors to object regions
does not improve hit rate on the hard pairs, or outline-anchored keypoints turn
out to be too unstable under viewpoint change to match at all. Both are
plausible and cheap to test.

## 3. Inputs and outputs

| | |
|---|---|
| Input | one image (RGB) |
| Output | keypoints `(N, 2)` with a boundary score each, and descriptors `(N, D)` |
| Matching | a separate stage: mutual NN with a ratio test today, learned later |
| Contract | `DetectorDescriptor.detect(image)`; `BenchmarkAdapter.match(reference, query)` for the harness |
| Optional | per-keypoint "on-object" probability, so downstream code can weight or reject |
| Not in scope | multiple instances, class labels, tracking |

## 4. Why boundaries, concretely

Boundaries have three properties that matter here:

1. **They belong to the object.** A boundary pixel is, by definition, where the
   instance meets something else. Interior texture may be shared with a similar
   object; the boundary shape is instance-specific.
2. **They constrain geometry.** Points spread along an outline span the object,
   so a similarity or homography fit is well-conditioned. Interior points
   clustered on one logo letter do not constrain scale.
3. **They are stable under appearance change.** Colour and illumination change
   across cameras; the silhouette does not.

The cost is real: boundary points are less repeatable than blobs under blur,
motion, and low resolution, and some boundaries are low-contrast. That trade-off
is the thing Phase 0 has to measure, not assume.

## 5. Where the boundary signal comes from

The four datasets give boxes, not masks. Three options, in order of cost:

1. **Classical pseudo-masks.** GrabCut initialised with the reference box, or
   edge-based segmentation, to get a rough object mask per crop. No new
   dependencies, works offline, noisy.
2. **Promptable segmentation.** A segmentation model (SAM or similar) prompted
   with the box, run once per crop, cached. Better masks, one new dependency,
   and it must be evaluated against the ROCm stack.
3. **Weak supervision without masks.** Train a boundary head on image gradient
   magnitude restricted to the reference box, with the box interior as a soft
   positive and far background as a hard negative. Cheapest, least precise.

Phase 0 uses option 1, because it needs to run this week.

## 6. Architecture options

A diagram of the implemented Stage A shape — frozen backbone, fused decoder,
two heads, extraction, and the adapter matcher — is in
[`src/osid/model/README.md`](../src/osid/model/README.md#architecture).

**A. Boundary-aware detector and descriptor, separate matcher.** A per-image
network that emits keypoints and descriptors, with matching left as a replaceable
stage: ratio-tested mutual nearest neighbours first, a learned matcher later.
Smallest change, directly testable against the current numbers, and it isolates
the claim: if A does not help, the idea is wrong for this data. Implemented in
`src/osid/model/` as `DetectorDescriptor`, with `BenchmarkAdapter` supplying the
matcher for the harness.

**B. Joint detector and descriptor.** A SuperPoint-style network whose detector
head is supervised on boundaries and whose descriptor head is trained on the
existing synthetic pairs. One model, better coupling, needs real training.

**C. End-to-end pair model.** Reference and query tokens through a
cross-attention encoder; heads for keypoint saliency, matching, and geometry.
The most "new model", the most data-hungry, and the hardest to attribute when it
fails.

The spec assumes **A first**, because it answers the question without a training
run, and **B** only if A's gains survive.

## 7. Training data, if it gets that far

Real pairs: FlickrLogos-32 (2,208 positive, 320 negative), PRW subset (5,881
pairs), RPC (1,244), HPatches (580) — far too little for a detector trained from
scratch, which is out of scope anyway. The plan is fine-tuning from a pretrained
backbone, plus synthetic pairs generated from the crops already on disk:

- random homography, scale, rotation, and crop;
- photometric change: brightness, contrast, blur, JPEG;
- occlusion by random patches, and background substitution behind the reference
  crop so the detector cannot rely on context;
- hard negatives: crops of other instances of the same class.

## 8. Phase 0: measurement before training

Four measurements, all without training, on the existing results and images:

1. **Where do current keypoints land?** For a sample of Flickr pairs, compute
   pseudo-masks and report the share of SIFT and SuperPoint keypoints inside the
   instance, on its boundary, and in the background.
2. **Where do the matches come from?** Same split for the correspondences that
   RANSAC keeps, versus the ones it rejects.
3. **Oracle region filtering.** Keep only keypoints inside the pseudo-mask
   (reference) and, as an upper bound, inside the GT box (query); re-run
   matching and measure hit rate on the 251 hard pairs. This is the direct test
   of the claim.
4. **Boundary-only filtering.** Keep keypoints within a band around the mask
   edge and repeat. This is the model's actual operating point, and the one most
   likely to be hurt by low repeatability.

Expected outcomes and what they mean:

| Outcome | Interpretation |
|---|---|
| Region filtering lifts the hard set meaningfully, boundary band holds up | Build A, then B |
| Region filtering helps but the boundary band collapses | Keep interior keypoints, drop the outline constraint; the claim narrows to "on-object" |
| Neither helps | The failures are not caused by background keypoints; stop and re-read the failure data |

## 9. Evaluation

The existing harness stays as-is, so results are comparable to the eight
matchers: hit rate and AP@0.5 on FlickrLogos-32 and PRW, MMA/HEA and corner error
on HPatches, with the fixed protocol.

Two metrics are added for the claim itself:

- **Outline precision**: share of predicted keypoints within *k* pixels of the
  instance boundary, on a human-checked subset.
- **Object-anchored match rate**: share of accepted matches whose reference
  keypoint is on the object, which is what the geometry actually depends on.

A win on hit rate with a loss on outline precision would mean the gain came from
something other than the stated idea, so both are reported together.

## 10. Milestones

| Step | Output | Gate |
|---|---|---|
| 0 | pseudo-masks for a sample, keypoint-on-object statistics | numbers exist and look sane |
| 1 | oracle region-filtering ablation on the hard pairs | decides whether to continue |
| 2 | boundary-band ablation | decides whether "outline" survives |
| 3 | architecture A implemented behind the existing `FeatureMatcher` interface | comparable to the eight matchers in `make bench` |
| 4 | training plan for B, with data generator | only if 3 shows a gain |

## 11. Risks

- **Boundary repeatability**: boundaries move under blur and low resolution; the
  detector may place points where the matcher cannot follow them.
- **Pseudo-mask quality**: GrabCut fails on cluttered backgrounds, which is
  exactly where the claim matters most.
- **Boxes are not silhouettes**: evaluating "on the object" with boxes conflates
  interior texture with the outline.
- **Data volume**: any learned detector needs synthetic pairs, and synthetic
  boundaries may not look like real ones.

Each of these is measurable in Phase 0, which is why Phase 0 exists.
