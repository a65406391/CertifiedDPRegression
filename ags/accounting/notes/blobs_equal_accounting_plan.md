# Blobs: fair accounting and repeated-release plan

## Status of the current comparison

`artifacts/blobs_cross_radius_pld_v3.json` is a diagnostic, not a fair headline
comparison.  AGS coordinates were composed from generic `(epsilon, delta)` PLDs,
whereas Gaussian DP-SGD used its mechanism-specific PLD.  Results from that artifact
must be labelled by the complete mechanism/accountant bundle and must not be presented
as an AGS-versus-DP-SGD result with matched accountant strength.

## Accountant-controlled comparisons

Run both of the following and keep them as separate panels.

1. **Matched generic control.** Reduce every coordinate release and every Gaussian
   release to a certified `(epsilon, delta)` pair, then compose both arms using the
   same pessimistic generic-PLD construction. This deliberately gives neither
   mechanism access to a mechanism-specific accountant.
2. **Matched mechanism-specific RDP.** Compose exact Gaussian RDP for DP-SGD and a
   worst-case smooth-Laplace RDP profile for AGS. The latter must pass the validation
   gate below before this panel is run. Do not compare generic AGS PLD with exact
   Gaussian PLD in this panel.

The two panels answer different questions: the first controls accountant strength;
the second compares the best validated accounting available to each mechanism inside
one common divergence framework.

## Repeated-release grid

Use the existing matched blob configuration: `N=200`, `b=20`, 200 optimizer steps,
learning rate `0.5`, coordinate clip `0.05`, nine parameters, and replace-one
adjacency. The initial online frequency screen evaluates every integer
`F in {1,...,10}`. Its predeclared first cell is one seed (`41`), total
`(epsilon, delta) = (10, 1e-5)`, contiguous MILP radii through `k=3`, and a
certified coordinate-GS tail for every omitted radius. The generic conservative
PLD is used for this screen; it is not mixed with Gaussian-specific PLD in a
method comparison.

| steps per AGS release | releases | corresponding MILP F |
|---:|---:|---:|
| 1 | 200 | 1 |
| 2 | 100 | 2 |
| 5 | 40 | 5 |
| 10 | 20 | 10 |

At every boundary, discard the clean parameter vector and start the next block from
exactly the released vector. The certificate for the next block is conditional on
that public released vector. Consequently, the old clean-trajectory boxes cannot be
reused after the first release: each subsequent block must be bounded from its actual
public start and fully serialized.

Use a staged seed count because this is solver-heavy: one seed for end-to-end
validation, five seeds for the screen, then 41 seeds only for predeclared finalists.
No configuration may be selected using test accuracy.

Repeated-release artifacts live under `ags/blobs/runs/repeated_release`,
not the clean-trajectory `runs/certified` tree. Each run separates restartable
released-state checkpoints from non-public certificate audit traces. The former
contains the released vector, RNG state, optimizer/schedule state, sample indices,
and accountant state. The latter contains the clean nominal, interval endpoints,
Gurobi formulations, and solve records, and is never used to initialize a later
block.

## Sampling and amplification

The current cyclic blob experiment has no random sampling event and receives no
amplification. In particular, an `F=10` block consumes `F*b=200=N` records, so even a
fresh without-replacement superbatch would have `q_block=1`.

An amplification study is a separate algorithmic variant. Use a larger private
population and sample one fixed-size superbatch of `F*b` records without replacement
per AGS block. Compose all nine coordinates, amplify the complete vector release once,
then compose blocks. Give DP-SGD the matched sampling protocol and apply its
mechanism-specific sampling accountant. Do not amplify individual AGS coordinates.

## Smooth-Laplace RDP validation gate

For adjacent datasets, normalize the two scalar output laws to

`P = Laplace(0, 1)` and `Q = Laplace(a, r)`.

Smoothness and local sensitivity imply

`r in [exp(-beta), exp(beta)]` and
`abs(a) <= (epsilon_coordinate/2) * min(1, r)`.

Before using a mechanism-specific accountant:

1. derive the scalar Renyi integral piecewise over the breakpoints `0` and `a`;
2. compare it with high-precision numerical quadrature over a dense grid of
   `(alpha, r, a)` values;
3. maximize both privacy directions over the full feasible set, not just sampled
   endpoints;
4. verify the optimizer against interval branch-and-bound or an analytic monotonicity
   proof;
5. compose coordinates and releases additively in RDP and convert once to
   `(epsilon, delta)`;
6. reproduce the theorem-level `(epsilon, delta)` guarantee as a conservative check;
7. fail closed on any infinite-RDP order or unresolved numerical maximization.

Only after this gate passes should the mechanism-specific AGS curve be compared with
Gaussian RDP.

## Required outputs

- achieved epsilon and delta under both matched-accountant panels;
- accuracy/loss distributions and runtime;
- per-release and cumulative privacy curves;
- smooth-sensitivity and noise-scale quantiles per release;
- selected maximizing radius for every coordinate and release;
- MILP status, dual-bound use, and serialization manifest for every certificate;
- explicit sampling rate and amplification location;
- a flag distinguishing fully certified results from diagnostics.
