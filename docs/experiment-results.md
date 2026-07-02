# NRC Experiment Results

This document records the reproducible experiment outputs for the current Moer NRC comparison branch.

## Branch and backends

- Branch: `feature/nrc-comparison`
- CPU backend: `RunningAverageRadianceCache`
- GPU backend: `TcnnRadianceCache` through tiny-cuda-nn
- Compared modes:
  - `path_low`: low-spp path tracing
  - `path_reference`: higher-spp path tracing reference
  - `nrc_cpu`: cache continuation estimator with CPU cache
  - `nrc_tcnn`: cache continuation estimator with TCNN cache
  - `two_level_cpu`: residual-corrected two-level estimator with CPU cache
  - `two_level_tcnn`: residual-corrected two-level estimator with TCNN cache

## Reproduction commands

```powershell
& cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake -S . -B build-tcnn -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DENABLE_TCNN_NRC=ON -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=86'
& cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 && cmake --build build-tcnn --config Release'
python scripts\run_experiment.py configs\experiments\testball_tcnn_matrix.json --executable target\bin\Moer_r.exe
python scripts\run_experiment.py configs\experiments\teapot_tcnn_matrix.json --executable target\bin\Moer_r.exe
python scripts\summarize_results.py results --output results\summary.csv
```

## Formal runs

- `results/testball_tcnn_matrix/20260703-010716`
- `results/teapot_tcnn_matrix/20260703-010802`
- Combined summary:
  - `results/summary.csv`
  - `results/summary.md`

## Key Results

| scene | method | spp | seconds | queries | train samples | train calls | residual samples | query s | train s | PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| testball | path_low | 2 | 0.3383 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 17.9257 |
| testball | path_reference | 16 | 0.3967 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | inf |
| testball | nrc_cpu | 2 | 0.3177 | 6478 | 6230 | 6230 | 0 | 0.0054 | 0.0042 | 11.1809 |
| testball | nrc_tcnn | 2 | 1.1183 | 6500 | 6235 | 24 | 0 | 2.5967 | 0.0311 | 10.0764 |
| testball | two_level_cpu | 2 | 0.3051 | 6475 | 6214 | 6214 | 6475 | 0.0056 | 0.0026 | 15.0122 |
| testball | two_level_tcnn | 2 | 1.0762 | 6490 | 6225 | 24 | 6490 | 2.4004 | 0.0288 | 11.2268 |
| teapot | path_low | 2 | 0.2470 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | 12.1613 |
| teapot | path_reference | 16 | 0.3351 | 0 | 0 | 0 | 0 | 0.0000 | 0.0000 | inf |
| teapot | nrc_cpu | 2 | 0.2615 | 18432 | 18432 | 18432 | 0 | 0.0288 | 0.0265 | 11.3785 |
| teapot | nrc_tcnn | 2 | 2.4690 | 18432 | 18432 | 72 | 0 | 13.6546 | 0.0429 | 11.0502 |
| teapot | two_level_cpu | 2 | 0.2629 | 18432 | 18432 | 18432 | 18432 | 0.0293 | 0.0230 | 11.0475 |
| teapot | two_level_tcnn | 2 | 2.4504 | 18432 | 18432 | 72 | 18432 | 11.1673 | 0.0440 | 10.9156 |

## Interpretation

The experiment proves that the TCNN backend is wired end to end: it builds, runs, receives training samples, performs training calls, produces cache queries, and writes HDR outputs and CSV metrics.

The current TCNN version is slower than the CPU validation backend because query execution is still one vertex at a time. The measured `nrc_query_seconds` dominates runtime, while batched training is already relatively small. This is the main engineering gap between this coursework implementation and a production NRC implementation.

The two-level estimator consistently records residual samples and changes the output. On `testball`, `two_level_cpu` improves PSNR over `nrc_cpu`; the TCNN two-level path is functional but still limited by the single-query GPU path.
