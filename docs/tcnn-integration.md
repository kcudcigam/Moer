# TCNN Integration Notes

The current experiment branch uses `RunningAverageRadianceCache` as a CPU cache backend to validate the NRC data flow.
This is not the final high-performance NRC backend.

## Local feasibility check

Detected on this machine:

- CUDA: 12.6
- GPU: NVIDIA GeForce RTX 3060, 12 GB
- Current successful Moer build: MinGW Makefiles
- MSVC `cl.exe`: not found in the active shell

This means TCNN is feasible at the hardware/CUDA level, but it should be integrated through a Visual Studio/MSVC CUDA build environment rather than the current MinGW build.

## Intended backend

Add `TcnnRadianceCache : INeuralRadianceCache` with the same public API already used by `NrcPathIntegrator`.
The backend should replace `RunningAverageRadianceCache` without changing integrator control flow.

Required implementation:

- Convert `RadianceQuery` to a flat feature vector.
- Batch `queryBatch()` calls instead of per-hit network calls.
- Batch upload `RadianceSample` targets.
- Train an MLP with encoded position/direction/material features.
- Use clamped or log-radiance RGB loss.
- Report `nrc_query_time` and `nrc_train_time`.

## Current experimental limitation

The generated experiment results compare:

- low-spp path tracing,
- high-spp reference,
- NRC control flow with a CPU running-average cache,
- two-level residual hook with a global residual corrector.

They prove the Moer-side NRC experiment framework and data path, but they are not final TCNN performance results.
