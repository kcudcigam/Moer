# NRC Framework

This branch adds the shared integration layer for Neural Radiance Cache experiments.

## Current status

- `renderer.integrator = "nrc_path"` selects the new path integrator.
- `renderer.nrc.mode = "path_trace"` keeps full path tracing and is used as a framework baseline.
- `renderer.nrc.mode = "nrc"` uses the cache continuation estimator after `query_bounce`.
- `renderer.nrc.mode = "two_level"` routes through the two-level estimator hook.
- The active cache backend is `DummyRadianceCache`; TCNN and residual correction are implemented on downstream branches.

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
      "query_bounce": 2
    }
  }
}
```

## Branches

- `feature/nrc-framework`: shared interfaces, configuration, and dummy backend.
- `feature/nrc`: TCNN-backed Neural Radiance Cache.
- `feature/two-level-mc`: residual-corrected cache estimator.

## Smoke test

The framework was validated with a temporary copy of `scenes/box` using:

```powershell
cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-mingw --config Release -j 4
target\bin\Moer_r.exe E:\tmp\moer_nrc_box
```

Both `mode = "path_trace"` and `mode = "nrc"` completed with `DummyRadianceCache`.
