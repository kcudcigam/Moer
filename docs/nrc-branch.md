# NRC Branch

This branch extends `feature/nrc-framework` with the first executable NRC data path.

## Implemented

- `NrcSampleBuffer` stores bounded training samples.
- `RunningAverageRadianceCache` is a thread-safe CPU cache backend used to validate data flow before TCNN.
- `NrcPathIntegrator` collects a path-traced continuation target at the NRC query point.
- The collected target is enqueued and used to update the cache during rendering.

## Next replacement

Replace `RunningAverageRadianceCache` with `TcnnRadianceCache` while keeping the same `INeuralRadianceCache` API.
The expected TCNN backend should:

- batch encode `RadianceQuery` features,
- batch upload `RadianceSample` targets,
- train with log-radiance or clamped RGB loss,
- query all queued continuation points in batches.

## Validation

The branch builds with:

```powershell
cmake --build build-mingw --config Release -j 4
```

Smoke test used a temporary 64x36, 1 spp copy of `scenes/box` with:

```json
{
  "renderer": {
    "integrator": "nrc_path",
    "spp": 1,
    "nrc": {
      "mode": "nrc",
      "query_bounce": 1,
      "train_steps": 1,
      "max_training_samples": 1024
    }
  }
}
```
