# GPU checks

Run `npm ci`, then `npm run dev`. Open these paths on the local server in a WebGPU-capable browser. The checks are separate from the application and are not included in its production build.

| Page | Coverage |
| --- | --- |
| `/check/import-check.html` | OBJ topology, unit-independent fitting, material paths, UV orientation, presentation isolation and scene-training gradients |
| `/check/import-check.html?large` | Includes the 2048 × 2048 lens dispatch regression |
| `/check/import-check.html?flood` | Includes a production-resolution scene session, painted stroke and restoration of the initial flood after clearing |
| `/check/optics-check.html` | Independent CPU image/flux comparisons, finite-difference gradients, output clearing, and resource contracts |
| `/check/optimizer-check.html` | Target normalization, loss and regularization gradients, Adam updates, and optimizer lifecycle |
| `/check/loss-filter-check.html` | GPU Gaussian-filter gradients against CPU finite differences, including boundaries and non-square images |
| `/check/integration-check.html?shape=circle` | Real A/B optimization, received power, stray light, stopping, clearing, warm restart, and final-preview consistency |
| `/check/a-study/?checks-only` | 12 edge cases across all four optical strategies |
| `/check/a-study/` | Extended checks and per-stage/full-step GPU benchmarks; requires `timestamp-query` support |

The optics, import, optimizer and integration pages report `ALL PASSED`. The filter page reports `passed: true`. Extended checks and benchmarks show completion in the page status. Also inspect the browser console for GPU validation errors.

The integration page also supports `shape=offset` and `shape=bar`, and `strategy=baseline`, `cached`, `cooperative`, or `striped`. These options belong to diagnostic pages; the application defaults to `baseline`.

## Interpreting the checks

- `optics-reference.ts` is an independent CPU polygon-integration reference used only by optical checks. It is not a production CPU solver.
- The optimizer check uses a small analytic forward model to isolate optimizer errors. The integration check uses the real optical forward and backward model.
- The presentation check captures the prism and refracted light from the side to verify that one Room switch toggles both. It also checks the debug views and compares the optical buffers byte for byte. Controlled beams check first-hit clipping and equal-length paths viewed from above, near the rim and from opposite ends. A ray crossing 48 thin layers should integrate to the same energy as a uniform field with the same mean. At fixed fluence, a narrow ridge must appear brighter than a broad sheet. Neighbouring light alone must leave empty voxels dark. The separate ribbon fixture covers the earlier line renderer.
- Debug irradiance uses a fixed logarithmic scale relative to incident irradiance, without Room smoothing, environment fill or drawing-dependent exposure.
- Splat candidates counts projected bounding rectangles before visibility/coverage rejection. It describes screen-space rendering workload and varies with camera and viewport resolution, rather than measuring received energy.
- Optical backward differentiates refraction, polygon clipping, and projected-area normalization, then gathers adjacent triangle derivatives into vertex gradients. Derivatives are piecewise; exact clipping topology changes need separate analysis.
- Invalid refraction/projection is skipped, and off-screen power is not redistributed. Check received power as well as loss; a darker background alone does not establish a better lens.
- Console loss is the unfiltered image loss. Early optimization uses a filtered residual, so the logged loss need not decrease on every step. The configured iteration limit is a budget, not a convergence proof.
- Integration checks use batches of five updates; the application uses `config.maxStepsPerFrame`. Diagnostic wall time is not UI convergence time or pure GPU kernel time.

## Performance measurements

The study page compares baseline, cached footprints, 64-lane cooperative work, and 8-lane work. It checks image/gradient agreement before timing and provides a raw JSON download. Record the GPU, browser, dimensions, and strategy with each measurement; software-adapter results do not establish hardware performance.

Keep downloaded measurements and generated plots outside the repository or under the ignored `check/a-study/results/` directory. Historical reports and one-off comparison results are not required to run these checks.
