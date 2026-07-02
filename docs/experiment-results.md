# NRC Experiment Results

This document records the reproducible experiment outputs for the current Moer NRC comparison branch.

## Branch and backend

- Branch: `feature/nrc-comparison`
- Backend: CPU `RunningAverageRadianceCache`
- Compared modes:
  - `path_low`: low-spp path tracing
  - `path_reference`: higher-spp path tracing reference
  - `nrc`: cache continuation estimator
  - `two_level`: cache estimator plus global residual correction

The current results validate the Moer integration, data collection, and comparison pipeline. They are not TCNN performance numbers.

## Reproduction commands

```powershell
cmake --build build-mingw --config Release -j 4
python scripts\run_experiment.py configs\experiments\teapot_matrix.json
python scripts\run_experiment.py configs\experiments\testball_matrix.json
python scripts\summarize_results.py results --output results\summary.csv
```

## Final runs used for the report

- `results/teapot_matrix/20260702-234333`
- `results/testball_matrix/20260702-234335`
- Combined summary:
  - `results/summary.csv`
  - `results/summary.md`

## Key results

| scene | method | spp | seconds | queries | train samples | residual samples | PSNR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teapot | path_low | 2 | 0.3152 | 0 | 0 | 0 | 12.1547 |
| teapot | path_reference | 16 | 0.3349 | 0 | 0 | 0 | inf |
| teapot | nrc | 2 | 0.2562 | 18432 | 18432 | 0 | 11.1779 |
| teapot | two_level | 2 | 0.2444 | 18432 | 18432 | 18432 | 11.2862 |
| testball | path_low | 2 | 0.2670 | 0 | 0 | 0 | 17.8322 |
| testball | path_reference | 16 | 0.3626 | 0 | 0 | 0 | inf |
| testball | nrc | 2 | 0.2701 | 6499 | 6265 | 0 | 12.6372 |
| testball | two_level | 2 | 0.2696 | 6492 | 6210 | 6492 | 14.1482 |

## Interpretation

The CPU cache is intentionally simple, so the image quality is not expected to match a TCNN NRC. The useful evidence from these runs is:

- Moer can switch between path tracing, NRC continuation, and two-level correction from config.
- NRC and two-level modes collect nonzero query and training samples on surface scenes.
- Two-level mode records residual samples and changes the prediction result.
- The scripts produce HDR images, quality metrics, timing data, and component-level NRC statistics.

For final high-performance claims, replace `RunningAverageRadianceCache` with a batched TCNN backend and rerun the same matrices.
