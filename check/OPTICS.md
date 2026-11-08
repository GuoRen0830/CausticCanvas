# Forward and backward optics checks

Run `npm ci`, then `npm run dev`. Open `/check/optics-check.html` on the local server in a WebGPU-capable browser. A successful run ends with `ALL PASSED`. This diagnostic page is separate from the application and does not implement B's optimizer.

## Implementation

- `src/optics/forward-model.ts` implements `encode` and the proposed optional `encodeBackward` contract.
- `src/optics/optics.wgsl` reconstructs face normals, refracts parallel light, projects each triangle, and clips it against receiver pixels. Pixel values are average irradiance in W/m².
- Forward accumulation uses compare-and-swap on f32 bit patterns stored in u32 atomics. The exported buffer is read as f32. This avoids fixed-point quantization but floating-point summation order is not deterministic.
- Backward uses forward-mode automatic differentiation local to each triangle: a scalar carries its value and three height derivatives. The clipping intersection and area normalization are differentiated as well as refraction. Each triangle immediately contracts these derivatives with the incoming pixel gradient. A second dispatch gathers adjacent triangle contributions into each height gradient, overwriting all elements without float gradient atomics.
- Parameters are immutable per configuration, so different configurations can be encoded before one submission. Grid or image resolution changes require a new forward-model instance. Heights may use different buffers of the same layout.
- The caller must consume a forward result and its backward pass before overwriting the heights or reusable outputs. The API cannot detect external writes to a borrowed GPU buffer.

## Validation

The checks compare independent CPU polygon integration with WebGPU results for flat, perturbed, equal-index, clipped, and total-internal-reflection cases. They compare all nine height derivatives on a small grid with CPU and GPU central differences, including a clipped footprint. They also check output clearing, zero-gradient overwrites, stale-result rejection, same-encoder parameter preservation, a loss-reducing optical gradient step, and the default 33 × 33 / 256 × 256 configuration with 0.04 W incident power. A default flat lens is checked for a finite gradient and a loss-reducing step toward a curved-lens target.

The CPU-stepped diagnostic gradient update is a derivative test, not a production optimization loop. B remains responsible for GPU updates and training state.

## Numerical scope

The derivative is piecewise analytic. At exact clipping topology changes, equality comparisons select a branch; a unique derivative may not exist and central differences may disagree. No smoothing or claim of global convergence is implied. Both the clipping numerator and projected-area denominator are differentiated on regular configurations.

Total internal reflection (`k <= 1e-8`), nearly horizontal output (`d.z <= 1e-6`), receiver intersections behind any corner, and non-finite projection produce an inactive beam. Projected areas at or below `1e-8` pixel² are omitted from receiver integration and have zero optical gradient. Valid beam flux is retained in the beam field for the latter case; received power can consequently be smaller than exported beam power. Invalid geometry needs regularization/constraints in B, not renormalization of the receiver image. Non-finite triangle gradients are replaced with zero as a safeguard.

This is a correctness baseline. CAS contention, large projected footprints, per-call binding creation, and differentiated clipping can be expensive. It does not implement tile binning or the room renderer. GPU timing and optional cooperative kernels are evaluated in [A-RESULTS.md](A-RESULTS.md). Software-adapter checks establish shader and numerical behavior, not hardware performance.


## A/B integration

B's `feat/lens-optimizer` implementation at `cb0878f` is connected through the existing optional backward interface. Production changes for this integration are limited to `src/config.ts`, `src/optimization/lens-optimizer.ts`, and `src/optimization/shaders/optimizer.wgsl`; A and the application scheduler are unchanged.

Open `/check/integration-check.html` to run a real optical target through A, B, and the shared simulation controller. This uses the default 33 × 33 lens and 256 × 256 receiver for 3,000 Adam updates, in diagnostic batches of five. It verifies loss reduction, finite heights, untouched borrowed input heights, same-version stopping, empty-target preservation, warm restart, and consistency of the final preview. The normal UI still uses one step per frame. Diagnostic wall time is not a kernel benchmark or the UI convergence time.

The first native Apple / Metal test reduced normalized image loss from 2.7023455 to 1.6042389 for a centered Gaussian target, about 40.6%. This establishes a working optical optimization loop, not a high-quality reconstruction guarantee. Initial iterations can increase loss. The configured 3,000-step stopping criterion is a budget limit rather than a convergence proof. More demanding drawings need separate quality evaluation and tuning.


## Coarse-to-fine loss

The optimizer now starts with a Gaussian-filtered residual, reducing sigma from 16 receiver pixels to zero over the first 80% of each run. The final 20% uses the original pixel loss. The optical image and uploaded/drawn target are unchanged. The backward chain applies the transpose filter as well: the data gradient is `2 K^T K(E-T) / (N * meanIrradiance^2)`, not simply a blurred target. Zero padding and a position-independent normalization make the separable filter symmetric, including at screen edges. Console loss remains the unfiltered image loss for comparison; it can rise while the coarse objective improves.

Adam's step size decreases quadratically to 5% of its initial value. The curvature weight is 10 instead of 0.1 to discourage fragmented surface slopes. A new target resets the schedule and Adam moments while preserving the current heights.

Native Apple / Metal comparisons used the same flat 33 by 33 lens, 256 by 256 receiver, and 3,000 updates for the original optimizer and the new optimizer. The centered circle radius was 0.06 m; the offset circle moved by (0.052, -0.032) m; the bar half-size was (0.092, 0.024) m. All fractions below are relative to incident power, not received power.

| Target | Original image MSE | New image MSE | Original outside power | New outside power | New received power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circle | 6.9175 | 0.4006 | 58.04% | 4.90% | 98.15% |
| Offset circle | 8.6203 | 1.9053 | 64.33% | 15.84% | 86.62% |
| Bar | 10.0351 | 0.4009 | 64.00% | 5.34% | 98.70% |

This improves these examples but does not remove all stray light or guarantee convergence. The offset circle still loses substantial received power; an improvement in background darkness alone is not sufficient evidence of a better lens. Floating-point accumulation makes repeated runs slightly different.

Run `/check/integration-check.html?shape=circle` for an optical regression that checks spill below 10%, received power above 95%, finite heights, stopping, clearing, and warm restart. `shape=offset` and `shape=bar` exercise the other targets. `/check/loss-filter-check.html` compares the actual GPU filter gradient with independent CPU finite differences at sigma 0, 1.5, and 4 on a non-square image, including boundary pixels. Maximum absolute derivative error was below 4e-9.
