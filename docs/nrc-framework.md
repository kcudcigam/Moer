# NRC Framework

This branch adds the shared integration layer for Neural Radiance Cache experiments.

## Current status

- `renderer.integrator = "nrc_path"` selects the new path integrator.
- `renderer.nrc.mode = "path_trace"` keeps full path tracing and is used as a framework baseline.
- `renderer.nrc.mode = "nrc"` uses the cache continuation estimator after `query_bounce`.
- `renderer.nrc.mode = "two_level"` routes through the two-level estimator hook.
- `renderer.nrc.backend = "cpu"` uses `RunningAverageRadianceCache`, a CPU backend used to validate Moer-side NRC data flow.
- `renderer.nrc.backend = "tcnn"` uses the tiny-cuda-nn backend when Moer is built with `ENABLE_TCNN_NRC=ON`.
- `renderer.nrc.mode = "two_level"` records residual samples and applies a global residual correction hook.
- TCNN build and runtime details are in `docs/tcnn-integration.md`.

## Scene configuration

Moer currently reads rendering options from the `renderer` object in each `scene.json`.
To run the framework on an existing scene, merge one of the JSON snippets from `configs/` into that scene's `renderer` object.

Example:

```json
{
  "renderer": {
    "integrator": "nrc_path",
    "spp": 16,
    "output_file": "results/nrc_dummy.png",
    "nrc": {
      "mode": "nrc",
      "backend": "tcnn",
      "query_bounce": 2,
      "train_steps": 1,
      "train_batch_size": 256
    }
  }
}
```

## Branches

- `feature/nrc-framework`: shared interfaces, configuration, and dummy backend.
- `feature/nrc`: original NRC-style cache continuation path.
- `feature/two-level-mc`: residual-corrected cache estimator.
- `feature/nrc-comparison`: combined comparison branch with experiment scripts, statistics, and result summaries.

## Smoke test

The framework was validated with a temporary copy of `scenes/box` using:

```powershell
cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-mingw --config Release -j 4
target\bin\Moer_r.exe E:\tmp\moer_nrc_box
```

Both `mode = "path_trace"` and `mode = "nrc"` completed with `DummyRadianceCache`.
