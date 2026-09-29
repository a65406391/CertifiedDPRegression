# Conventions

How this repository is organised, where new work goes, and what must not change. Written for a
collaborator adding an experiment, a dataset, a figure or a
baseline. `README.md` says what the methods are; this file says how the code is arranged and why.
Where the two disagree, this file wins for anything about layout and naming.

The three rules that matter most, in case nothing else is read:

1. **The AGT packages are not edited.** `abstract_gradient_training/` and
   `abstract_gradient_training_opt/` are used, never changed. Extensions go in `ags/agt_ext/`
   (additive) or `common/agt_patches.py` (a context manager that installs and removes a patch);
   every deviation from upstream is accounted for in `Notes/PACKAGE_CHANGES.md`.
2. **Stored evidence is never overwritten.** Certified bounds, results JSON, accounting artifacts
   and stored arrays are written once. A re-run gets a new tag or filename and is compared against
   the old with `tools/compare_results.py`.
3. **Nothing is done until the gate says so.** `PYTHONPATH=. python -m tools.reproduce_figures --check`
   must report `0 changed` and both test suites must pass before a change is called finished, and
   the commit message states those numbers.

## Running anything

Everything runs from the repository root as a module, with the root on `PYTHONPATH`:

```bash
PYTHONPATH=. python -m agtr.california.release --variant relu_h64_clamp_gamma0.1 --plot-only
PYTHONPATH=. python -m ags.blobs.plot_headline
PYTHONPATH=. python -m common.figures                     # lists every named figure
```

Never edit `sys.path` and never run a file by path. Every module imports without side effects: no
figure, computation, download or argument parsing happens at import time. Scripts put their body
under `main()` behind `if __name__ == "__main__":`, and parse arguments inside it. Importing any
module in the tree must be instant and silent.

Verification, in full:

```bash
PYTHONPATH=. python -m tools.reproduce_figures --check      # every registered figure, byte-compared
pytest tests/                                                # the two AGT packages
PYTHONPATH=. pytest -q common/tests common/accounting/tests \
    ags/blobs/test_ss_reference.py ags/mnist/test_ss_reference.py \
    ags/agt_ext/test_deep_per_sample.py                     # shared modules, accounting, bit-equality
ruff format <paths you touched> && ruff check --select F401,F821 ags agtr common tools
```

The project interpreter and environment caveats are in `Notes/REPO_NOTES.md` §8. Gurobi is needed
only to compute new certificates; every figure redraws from stored results without it.

## The map

```
common/           shared mathematics and machinery, used by both halves
common/accounting/  the accounting engine (calibration, PLD, RDP, sensitivity, config, immutable writes)
ags/              private learning: one directory per dataset, the study named beneath it
agtr/             private prediction: one directory per dataset, the four pipeline stages
tools/            the figure gate, the results comparator, the baseline they check against
tests/            the test suite of the two AGT packages (not ours; kept passing)
Notes/            the written record: pipeline diagrams, repository and methodology notes, logs
archive/          retired code and results, kept for provenance, never imported
```

| I want to add... | It goes in |
|---|---|
| a certificate, mechanism, metric or accountant used by more than one experiment | `common/` (`certificate.py`, `mechanisms.py`, `metrics.py`, `common/accounting/`) |
| a baseline (DP-SGD variant, PATE variant, objective perturbation) | `common/baselines.py` |
| an AGS ablation on a new dataset | `ags/<dataset>/ablations/` |
| accounting for a dataset | `ags/<dataset>/accounting/` |
| an AGT-R experiment on a new dataset | `agtr/<dataset>/` with `certify.py train.py release.py plot.py` |
| a figure for the paper | a `plot_*.py` beside the results it reads, registered in `tools/reproduce_figures.py` |
| a one-off check that answered its question | `<dataset>/probes/` with the banner (see Retiring) |
| an extension to the AGT packages | `ags/agt_ext/` or `common/agt_patches.py`, and a line in `Notes/PACKAGE_CHANGES.md` |
| a note about a result, a decision or a defect | `Notes/`, or a `RUNLOG.md` beside the stored values it describes |

## One directory per dataset, the study beneath it

`ags/` and `agtr/` are organised on a single axis at each level: first the dataset, then the study.
The dataset name appears once, as the folder, and is never repeated in a file name inside it.

```
ags/mnist/                     the certified lineage at the root: ags.py baselines.py shared.py ...
ags/mnist/ablations/           config.py run_*.py            runs/ figures/
ags/mnist/accounting/pca/      config.py run_*.py            ../artifacts/
ags/mnist/accounting/screens/  run_experiment.py run_*.py    configs/
ags/mnist/probes/              finished one-offs
agtr/california/               certify.py train.py release.py plot.py config.py
agtr/california/pate/          train.py release.py           the matched PATE comparison
agtr/california/values/<tag>/  the stored cells one campaign produced
```

What is shared by every dataset lives one level up, and nowhere else: mathematics in `common/`, and
the few accounting things that belong to no dataset in `ags/accounting/` (the accountant audit
sweep, the cross-dataset summaries and status plotter, `paths.py`). A new `ags/accounting/<thing>`
must be dataset-independent; if it reads or trains on one dataset it belongs under that dataset.

## Naming

**Scripts** carry a verb that says what they do; **libraries** are nouns.

| Name | Meaning |
|---|---|
| `certify.py` | the library one dataset's certified training shares: model, loader, one certified cell |
| `train.py` | produces certified cells into `values/<tag>/` (agtr) |
| `release.py` | turns stored cells into a release: certificate, staircase, noise, metrics, `results.json` |
| `plot.py` | draws from `results.json`, computes nothing |
| `run_<name>.py` | any other script that produces stored results |
| `plot_<name>.py` | any other script that only draws |
| `summarize_<name>.py` | reads stored results and writes a report (JSON and Markdown) |
| `config.py` | the constants and dicts the drivers beside it read |
| `shared.py` | the data, model, parameter packing and run logging one lineage shares |
| `<noun>.py` | a library: `opt_bound.py`, `losses.py`, `serialize.py`, `exact_linear_certifier.py` |

A name never repeats the dataset (`agtr/california/train.py`, not `train_california.py`) and never
carries a date or a version (`_v2`, `_final`); a campaign's tag or date goes on the folder its outputs
land in. `common.py` is not used as a file name because it collides with the `common/` package.

**In code**, the vocabulary of the paper is kept exactly: `A[k]` is the cross-radius certificate,
`W[k]` an envelope width, `SS` the smooth sensitivity, `GS` the global sensitivity, `beta` the
smoothing parameter, `k` a substitution radius, `F` a concretization frequency, `P` the parameter
count, `gamma` the clipping threshold. Width and radius are different numbers (the width inflates
sensitivity by 1.4 to 1.8 times) and were once the same variable name; that is how a defect survived
for months. Do not reuse a name for a different quantity.

Release modes are the keys of `common.mechanisms.RELEASE_MODES`; use them rather than spelling a
mechanism's name in a new place.

## Where outputs go

| Folder | Holds | Rule |
|---|---|---|
| `runs/` | stored runs of the AGS lineages and ablations | written by drivers, read by `plot_*` |
| `artifacts/` | immutable accounting evidence | written only with `write_immutable_json`; a rerun chooses a new name |
| `values/<tag>/` | the cells one AGT-R campaign produced: one `.npz` per cell, `config.json`, `manifest.json`, `RUNLOG.md` | a cell absent from the manifest was never completed |
| `bounds/`, `checkpoints/` | unrolled MIP bounds and the model checkpoints they need | inputs to certification, kept |
| `figures/` | the figures the scripts beside it draw, PNG only | regenerated by the gate; never edited by hand |
| `data/` | raw or cached inputs | never results |
| `reference/` | pinned published runs | read-only |

A stored file is evidence. It is not overwritten, renamed in place, or "fixed". When a result is
superseded the new run gets a new tag or filename, the old stays, and the `RUNLOG.md` or `notes/`
beside them says which is current and why. An invalid result is marked, not deleted: see the
`INVALIDATED_*.md` files under `ags/mnist/accounting/artifacts/` for the form.

Ablation outputs are routed by `common.paths.get_output_dir(dataset, experiment)` to
`ags/<dataset>/ablations/runs/<experiment>/`, and their figures default to
`ags/<dataset>/ablations/figures/`. Accounting artifacts are found by
`ags.accounting.paths.artifact(name)`, which routes on the dataset prefix every artifact name carries
(`blobs_*`, `pca_*`, `logistic_*`, ...).

**Every new results writer records provenance.** `common.provenance.stamp(config=...)` returns the
block: commit, whether the tree was dirty, time, interpreter, package versions, the configuration.
Store it beside the numbers. `tools/compare_results.py` ignores these fields when it compares two
result files, so provenance never makes a re-run look different.

## Configuration

Constants and dicts live in a `config.py` beside the drivers that read them, never inside a driver,
so that seven sibling scripts read the same experiment and a new dataset copies a config rather than
a driver. YAML is used in one place, the config-driven screens runner
(`ags/mnist/accounting/screens/run_experiment.py`, schema in `common/accounting/config.py`), and
should not spread elsewhere.

When a command-line flag overrides a config value, default the flag to `None` and resolve it as
`CFG["x"] if args.x is None else args.x`. The shorter `args.x or CFG["x"]` silently discards an
explicit `0` or `0.0`, and did so in six files before it was caught.

## Figures

A plot script draws from stored results and computes nothing. If a number is needed on a figure it
is computed by the driver and stored, then read.

Every figure has a producer registered in `tools/reproduce_figures.py`:

```python
Producer("agtr.california.release.relu", "agtr.california.release",
         ["--variant", "relu_h64_clamp_gamma0.1", "--plot-only"], note="the working configuration"),
```

`--check` runs every producer and compares each output to `tools/figure_baseline.json` byte for byte
(PDF dates stripped). A new figure shows as `not in baseline`; a changed one as `changed`. Record a
new or intentionally changed figure with `--save-baseline` only after the change has been explained
in the commit. When figures move without changing, prove it first (hash the old and new paths), then
re-baseline. `-k <substring>` runs a subset, `--list` names them, `--heavy` includes the slow ones.

Colours and style come from `common/plotting.py` (`C_AGS`, `C_DPSGD`, `C_PATE`, `C_NOMINAL`,
`C_CEILING`, `BLUES`, `log_eps_axis`, `apply_style`). A figure does not define its own palette.
Plot scripts must seed anything random they draw (jitter, sampling), or the gate cannot check them.

## Tests

The AGT packages' own suite is `tests/` and stays passing as shipped. `tests/test_budget_fix.py` is
a deliberately failing specification of a fix that has not been made; leave it failing.

Our tests live beside what they test: `common/tests/`, `common/accounting/tests/`, and a
`test_<module>.py` next to a single module (`ags/blobs/test_ss_reference.py`,
`ags/agt_ext/test_deep_per_sample.py`). A new shared module gets a test in `common/tests/`; a new
accountant gets one in `common/accounting/tests/`, which also holds the brute-force oracle it is
checked against.

## The pinned reference

`common/ags.py` is the reference AGS implementation and is pinned. `ags/blobs/ss_reference.py` and
`ags/mnist/ss_reference.py` are line-by-line ports of its release path, and their tests cite line
numbers inside it. These files, and `ags/mnist/release.py`, are excluded from formatting in
`pyproject.toml` for that reason. They are changed together or not at all, and a change re-runs the
two `test_ss_reference.py` suites. Do not "clean up" `common/ags.py`.

## Retiring code

Three destinations, each with a meaning:

- `<dataset>/probes/`: a finished one-off that answered its question. It stays runnable and carries
  this banner as its first lines:
  ```python
  # FINISHED PROBE -- kept for provenance; its conclusion is written up in <where>.
  # Not part of any current figure.
  ```
- `legacy/`: a superseded driver whose figure is still reproduced by the gate. Its module header names
  what replaced it.
- `archive/`: dead. Moved with `git mv`, never imported, excluded from formatting and from the gate.
  `original/` folders inside the live trees are frozen producers of the published December 2025
  figures: not imported, not edited.

Deleting is for code that has no result behind it. Anything that produced a stored number is moved,
not deleted, so the number stays explicable.

## Documentation

One `README.md` in `ags/`, one in `agtr/`, one at the root, and no others in the code trees. A new
experiment is documented by a section in its tree's README, not by a new file beside it. The module
header (the docstring at the top of every script) says what the module does, why it exists, how it is
run and where it writes. Longer prose goes to `Notes/`: `REPO_NOTES.md` for conventions of the
methods and the list of results that must not be reused, `PIPELINE.md` for the diagrams,
`REFACTOR_LOG.md` for what changed in the tree and when, `PACKAGE_CHANGES.md` for every departure
from upstream AGT. Notes that describe one campaign's stored values live beside them as `RUNLOG.md`.

Mermaid diagrams pin `color:` alongside `fill:` in every `classDef` so they read in a dark theme.

## Shell scripts

An orchestration script resolves the repository root through git, exports it as `PYTHONPATH`, uses
`${PYTHON:-python}`, and waits on a `MANIFEST.json` or an output file rather than on time:

```bash
set -euo pipefail
repo_root="${AGT_REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}"
cd "$repo_root"; export PYTHONPATH="$repo_root"
while [[ ! -s "$sentinel" ]]; do sleep 30; done
${PYTHON:-python} -u -m ags.blobs.accounting.run_rdp_comparison --output ags/blobs/accounting/artifacts/<new_name>.json
```

## Figures are PNG only

Every plotting module writes one `.png` per figure and nothing else (25 September 2026). No PDF
companions: they doubled the file count, carried a creation time that made byte comparison
impossible, and were never the file anyone looked at. `common.plotting.save_figure` and every
`plot_*.py` follow this; a new plotter must too, and `tools/reproduce_figures.py` only watches
`.png`. If the paper needs a vector figure, render it on the spot with `fig.savefig("x.pdf")` outside
the repository; do not commit it.

## Recipes

**Add a dataset to the AGS ablations.** Create `ags/<dataset>/ablations/` with `__init__.py`,
`config.py` (one dict per experiment) and `run_<experiment>.py` drivers built on
`common.ags.train_ags` and `common.baselines`, writing through `common.paths.get_output_dir`. Add the
dataset to the family pairs in `tools/reproduce_figures.py` so `common.figures` draws it, run
`--check`, and `--save-baseline` once the new figures are explained. Copy `ags/mnist/ablations/` as
the template.

**Add accounting for a dataset.** Create `ags/<dataset>/accounting/` with `config.py` defining the
experiment (data, learner, `PRIVACY_STEPS`, `EPSILONS`), `run_<name>.py` drivers that call the engine
in `common/accounting/` and write with `write_immutable_json` into `artifacts/`, and give every
artifact name the dataset prefix. Add that prefix to `ags.accounting.paths`. Copy
`ags/blobs/accounting/` as the template; its `config.py` documents why `PRIVACY_STEPS` is the epoch
count. For a config-driven screen on a new dataset, `run_experiment.py` first needs its MNIST loader
parameterised.

**Add an AGT-R dataset.** Create `agtr/<dataset>/` with `config.py`, `certify.py` (loader, model,
`train_cell`), `train.py` (the sweep, writing `values/<tag>/`), `release.py` (certificate from
`common.certificate`, scales from `common.mechanisms`, metrics from `common.metrics`, `results.json`
with provenance) and `plot.py`. Register the plot. `agtr/california/` is the template; the dense
radius battery matters because a gap in `k` costs `e^{beta*gap}` in smoothness, and `release.py`
refuses gaps unless `--allow-gaps` is passed for a saturated battery.

**Add a mechanism or baseline.** A mechanism is an entry in `common.mechanisms.RELEASE_MODES` with
its scale function beside the others and a test in `common/tests/test_certificate.py` or a new file
there. A baseline is a function in `common/baselines.py` matched to the AGS trajectory (same data,
same clipping, same steps), with the matching stated in its docstring.

**Change a stored result.** Do not. Re-run into a new tag or name, compare with
`PYTHONPATH=. python -m tools.compare_results old.json new.json`, and write down in the `RUNLOG.md` or
`notes/` which is current and why.

## Definition of done

- The gate reports `0 changed`, or every change is intentional, explained, and re-baselined.
- Both test suites pass; `ruff format` and `ruff check --select F401,F821` are clean.
- Every module you touched imports silently.
- New results carry `common.provenance.stamp()`; nothing stored was overwritten.
- The tree's README section and, for a structural change, `Notes/REFACTOR_LOG.md` are updated.
- The commit message says what was verified and with what numbers.

## History that is not a convention but will bite

`Notes/REPO_NOTES.md` §7 lists results that must not be reused. In one line each: the paper-era AGS
runs under `ablations/runs/` omit the per-coordinate allocation factor `P` and are inputs for the
plotting pipeline only; the paper's stored MNIST run is under-noised by exactly `P = 650`; the legacy
`common/sensitivity.py` routines return a width where the paper's `A^k` is a radius, which is why the
retired California driver cannot reproduce Figure 2 and `agtr/california/repro/` can; the full-batch
PATE MILP (`train_milp_unsound.py`) is named for what a single-substitution attack showed about it.
