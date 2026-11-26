# benchmark/

Everything that measures. This package is the harness: it knows how to run a
matcher over a manifest and how to score the result. It does not contain the
model under development — that lives in [`../model/`](../model/).

## Files

| File | Responsibility |
|---|---|
| `cli.py` | argument parsing, the box and HPatches entry points, matcher construction |
| `runner.py` | run one matcher over a manifest, produce a model report |
| `registry.py` | `MODELS`: the eight baselines, and where a new model is registered |
| `geometry.py` | IoU, box projection through a homography, crop and image loading |
| `metrics.py` | average precision, per-model summary (hit rate, IoU, stages, latency) |
| `reports.py` | `results.md` and the HPatches report |
| `serialization.py` | per-sample record → JSON |
| `types.py` | `SampleResult`: one row of a report |
| `gpu.py` | device configuration, peak memory accounting, cleanup |

## Subpackages

| Path | Contents |
|---|---|
| `matchers/` | the eight baselines: SIFT/ORB/AKAZE, SuperPoint/ALIKED + LightGlue, LoFTR, RoMa, FlexGlue |
| `matchers/*/_vendor/` | vendored upstream inference code, kept under its original licence |
| `validation/` | HPatches sequence loading and MMA/HEA/corner-error metrics |

## Adding a model

The harness only requires one method, so anything that implements it can be
benchmarked next to the baselines:

```python
class Model:
    def match(self, reference: np.ndarray, query: np.ndarray) -> MatchResult: ...
```

`MatchResult` is in `matchers/base.py`. Register the class in `registry.MODELS`
and it works in both protocols without further plumbing.
