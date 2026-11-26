# model/

The model under development. Empty by design: week 4 is the first week that
builds it.

## Contract

The benchmark must be able to run it like any baseline, so the model implements
the same interface the matchers do:

```python
class Model:
    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult: ...
```

with `MatchResult` from `benchmark.matchers.base`. That is what makes it
comparable: same manifest, same protocol, same reports, registered once in
`benchmark.registry.MODELS`.

## What it will be

Given a reference crop and a query image, predict keypoints on both and the
correspondences between them, with keypoints on the **outline of the object**
rather than anywhere the detector finds contrast.

The design, the falsification conditions, and the measurements that gate
training are in [`docs/model-design.md`](../../../docs/model-design.md).

## Layout

```
model/
├── README.md      this file
├── __init__.py    package marker
├── detector.py    boundary-aware keypoint head (not written yet)
├── descriptor.py  descriptors and matching (not written yet)
└── geometry.py    similarity/box prediction (not written yet)
```

Those files are placeholders for the design, not commitments: Phase 0 decides
whether the boundary premise survives before any of them is written.
