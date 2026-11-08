# A to B Handoff: Forward and Backward Optics

## Scope and status

A implements the optical simulation and its height derivatives within the existing CausticCanvas framework. The implementation follows the physical model and data conventions in documents 01 and 02, and the backward interface proposed in document 03.

Implemented:
- Forward simulation: triangle normals, Snell refraction, receiver projection, and triangle/pixel overlap energy integration.
- Backward computation: propagate incoming pixel derivatives through the optical model and gather contributions at shared height vertices.
- Independent CPU comparisons, finite-difference gradient checks, boundary tests, and GPU performance experiments.
- Integration with B's existing Adam optimizer in the local project.

The `gradient descent first, then L-BFGS` pipeline has not been implemented. L-BFGS remains a possible B-side extension. The current full-loop implementation uses Adam.

## Files to merge

| File | Purpose |
| --- | --- |
| `src/optics/forward-model.ts` | Factory, resources, validation, forward/backward command recording |
| `src/optics/optics.wgsl` | Baseline optical computation and derivatives |
| `src/optics/cooperative.wgsl` | Optional cooperative kernels; required by the factory import even with the baseline |
| `src/contracts.ts` | Merge the `encodeBackward` declaration if missing; preserve unrelated contract changes |

Recommended standalone checks:
- `check/optics-check.html`
- `check/optics-check.ts`
- `check/optics-reference.ts`

The performance report is `check/A-RESULTS.md`; its figures and recorded data are under `check/a-study/results/`. The extended test and benchmark harness is under `check/a-study/`. Full optimizer integration checks are in `check/integration-check.html` and `check/integration-check.ts` and depend on the corresponding B implementation.

Do not copy `node_modules/`, `dist/`, or `.Rhistory` as part of the source handoff. No new application dependencies are required. Review changes before merging over concurrent work.

## Factory and interface

Create the model once using the application's shared GPUDevice:

```ts
const forward = await createForwardModel(device, initial)
if (!forward || !forward.encodeBackward) {
  throw new Error('Forward and backward optics are required.')
}
const optimizer = await createLensOptimizer(device, initial, forward)
```

This excerpt illustrates the existing connection; it is not a replacement for the application scheduler. The default factory strategy is `baseline`.

```ts
interface ForwardModel {
  encode(encoder: GPUCommandEncoder, input: ForwardInput): ForwardResult

  encodeBackward?(
    encoder: GPUCommandEncoder,
    input: ForwardInput,
    result: ForwardResult,
    pixelGradient: GPUBuffer,
    heightGradient: GPUBuffer,
  ): void

  destroy(): void
}
```

`encodeBackward` remains optional in the shared type for compatibility with forward-only implementations. This A implementation provides it.

## Responsibilities and command order

```text
A forward: current heights -> receiver irradiance E
B loss: E and target T -> loss and pixelGradient = dL/dE
A backward: pixelGradient -> heightGradient = dL/dh
B regularization: add constraint/regularization derivatives
B optimizer: update heights
Repeat
```

A does not compute the target normalization, total loss, regularization, or optimizer update. B owns these operations.

Each step must use B's current working height buffer in `ForwardInput`. Pass the exact `ForwardResult` returned by the matching forward call to backward. Complete B's loss pass before calling backward. All methods record commands between passes and close their own passes. The application owns `finish()` and `submit()`; A does not submit commands or wait for readback.

Run forward again after the final height update when displaying the updated lens. The pre-update image is not a preview of the new heights.

## Buffer and lifetime contract

| Data | Contract |
| --- | --- |
| Heights | Caller-owned, row-major f32, one value per vertex; absolute Z in meters; shared usage `STORAGE | COPY_SRC | COPY_DST` |
| Receiver image | A-owned reusable buffer; `result.caustic.kind === 'buffer'`; row-major f32 irradiance in W/m^2; row 0 is the bottom row |
| Pixel gradient | B-owned storage buffer; at least `receiver.width * receiver.height * 4` bytes; one dL/dE per pixel; A reads it |
| Height gradient | B-owned storage buffer; at least `lens.columns * lens.rows * 4` bytes; one dL/dh per vertex, in loss per meter; A overwrites every element |
| Beams | A-owned buffer, one vec4<f32> per triangle; xyz unit propagation direction, w flux in W |

A overwrites height gradients rather than accumulating into prior results. B does not need to clear the buffer before backward. Add B's regularization gradients after A's backward pass.

All resources must use the same GPUDevice. Height-gradient storage must not alias input heights, pixel-gradient storage, or A's output buffers.

A reuses its output buffers. Consume the image and run backward before another forward evaluation overwrites the buffers and before the matching input heights change. JavaScript object copies do not preserve GPU data. Submit command buffers in the same logical order. The latest-result check cannot detect external writes to a height buffer.

Recreate the model and matching optimizer resources when grid or receiver resolution changes. Other optical parameters must match between forward and backward. Release owned resources only after their queued work is no longer using them.

Current application defaults are 33 x 33 lens vertices and a 256 x 256 receiver image. These are independent configuration values, not hard-coded algorithm dimensions. They correspond to 1,089 height values, 2,048 triangles and 65,536 image pixels.

## Switching B from Adam to L-BFGS

The optical interface is optimizer-independent. B can implement L-BFGS without changing the forward/backward signatures, provided it preserves the buffer and evaluation-order contract.

B must implement:
- L-BFGS history and search-direction computation.
- Objective and gradient evaluation including regularization.
- Line search and trial-height management.
- Acceptance/rejection of trial steps, surface constraints, and stopping rules.

For every trial height field, run a matching forward evaluation. If a gradient is needed, compute the matching pixel gradient and call backward before another trial overwrites A's buffers. Save gradients or other values that L-BFGS must retain in B-owned storage. A's reusable outputs are not optimizer history.

A rejected trial requires restoring or retaining the accepted heights correctly. If the latest forward result belongs to a rejected trial, recompute forward for the accepted lens before displaying it or requesting its gradient.

A does not choose how B performs line-search decisions. A CPU-controlled line search may introduce GPU readback or synchronization; preserving a fully GPU-resident iteration requires additional B-side design. Fixed-step Adam's current scheduling should not be assumed sufficient for L-BFGS.

## Local changes to B: review separately

The local integrated project also contains changes to B's implementation in:
- `src/optimization/lens-optimizer.ts`
- `src/optimization/shaders/optimizer.wgsl`
- `src/config.ts`

These changes add a coarse-to-fine Gaussian-filtered residual loss, a decaying learning rate, and stronger smoothness regularization (weight changed from 0.1 to 10). The filter width decreases to zero; this affects the training loss rather than blurring the displayed result. The update method is still Adam.

The A-only ZIP excludes those B/configuration files. If pushing the entire local source tree, those changes will be included unless separated beforehand. Review them with B rather than replacing B's work without comparison. Full-loop convergence and A+B timing results in the local report used this adjusted optimizer; they are not guaranteed for B's original implementation.

## Validation after merge

1. Install the existing project dependencies if needed and run `npm run build`.
2. Run `npm run dev` and use the address printed by Vite.
3. Open `/check/optics-check.html`. Expected result: `ALL PASSED`, with no WebGPU validation errors.
4. If the extended harness was merged, open `/check/a-study/?checks-only` and confirm all 48 scenario/strategy combinations pass.
5. Open the main page, draw a small centered circle, and check that optimization runs and a consistently defined loss decreases over sufficient iterations. Individual Adam steps need not be monotonic.
6. Add a stroke and check that the current lens is reused. Clear the target and check that updates stop. Verify finite heights and a preview matching the final heights.

The handoff review reran the standalone and extended GPU checks successfully and verified the A-only archive against the local source. A subsequent clean checkout passed `npm ci`, `npm run build` (TypeScript checking and Vite production build), and `git diff --check` before handoff. B should rerun the checks after merging into their branch.

## Known limits

The model assumes uniform parallel illumination and a planar normal for each lens triangle. It omits Fresnel reflection and absorption. Invalid/inactive contributions are skipped, and clipped power is not renormalized onto the receiver. Gradient contributions from valid adjacent triangles are retained.

Overlap derivatives are piecewise derivatives of the discrete forward model; exact clipping topology changes are not guaranteed smooth. Passing finite-difference tests does not guarantee exact reconstruction of every target.

Optional strategies are `cached`, `cooperative` and `striped`. Their performance is workload- and device-dependent; keep the baseline for initial integration. Receiver-tile binning/local accumulation is not implemented. L-BFGS and the paper's complex initialization are not implemented.
