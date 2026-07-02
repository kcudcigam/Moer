# TCNN Integration Notes

This branch contains a working tiny-cuda-nn backend for the Neural Radiance Cache experiment path.

## Build

The TCNN backend is optional and is enabled with `ENABLE_TCNN_NRC`.
On this Windows machine the reliable configuration is NMake from the Visual Studio developer environment:

```powershell
& cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake -S . -B build-tcnn -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DENABLE_TCNN_NRC=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=86'
& cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build build-tcnn --config Release'
```

Verified environment:

- CUDA 12.6
- NVIDIA GeForce RTX 3060, 12 GB
- tiny-cuda-nn commit `749dd70c5afc5a9dadb85e5652ed65d55e0ba187`

## Configuration

Use the backend from scene JSON or an experiment matrix:

```json
{
  "renderer": {
    "integrator": "nrc_path",
    "nrc": {
      "mode": "nrc",
      "backend": "tcnn",
      "query_bounce": 1,
      "train_steps": 1,
      "train_batch_size": 256,
      "max_training_samples": 16384
    }
  }
}
```

`backend = "cpu"` keeps the running-average validation backend. `backend = "tcnn"` creates `TcnnRadianceCache`.

## Implementation

The TCNN cache currently maps a `RadianceQuery` to 12 scalar features:

- position
- surface normal
- outgoing direction
- roughness
- material type
- bounce index

The network predicts RGB continuation radiance. Training targets are produced by the same continuation path used by the CPU NRC framework.

Current speed-oriented changes:

- TCNN is isolated behind `INeuralRadianceCache`.
- TCNN is created through `RadianceCacheFactory`.
- Training is triggered every `train_batch_size` collected samples.
- Single-query GPU buffers are reused to avoid per-query device allocation.
- Statistics include `train_calls`, query time, train time, target trace time, sample counts, and residual counts.

## Current limitation

The backend is end-to-end functional, but it is not yet paper-level fast. Moer currently queries the cache one path vertex at a time, so TCNN inference still performs many small synchronized GPU calls. The formal results show this clearly: training calls are batched, but `nrc_query_seconds` dominates TCNN runtime.

The next performance step is to restructure `NrcPathIntegrator` to gather continuation queries across pixels/tiles and call `queryBatch()` once per batch.
