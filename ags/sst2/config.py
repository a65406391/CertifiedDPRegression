"""Settings of the SST-2 hybrid experiment (DP-LoRA fine-tuning of all-mpnet-base-v2, then an AGS head).

One task per encoder: `sst2` is the frozen encoder (the IMDB recipe: PCA-8 of the pooled features, a
logistic head under contraction, P = 18); `sst2_ft_eps<e>` is the encoder fine-tuned with DP-LoRA at
(e, 5e-6) by ags/sst2/finetune.py, whose adapted features the same head is trained on. The frozen task
carries the lambda/gamma sweep (stage 1, T = 50) and the DP-SGD head baselines; the fine-tuned tasks pin
its finalist and certify once at T = 300 (stage 1 is the finalist itself, so the sweep and the final
certification are the same key). The head's budget grid is dense so that any split eps_1 + eps_2 of a
total budget can be read off; the head takes delta = 5e-6, the encoder's half of the pipeline's 1e-5.
"""

from ags.sst2.shared import load_task

_RADII = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
HEAD_EPS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 7.5, 8]  # every eps - eps_1 the splits need
FINETUNE_EPS = [0.25, 0.5, 1, 2, 3, 5, 8]
# the eps_1 = 0.25 encoder exists for the split (0.25, 0.25) at a total of 0.5; only its task releases the
# head at 0.25 (the stored tasks keep HEAD_EPS, so their headlines are unchanged)
HEAD_EPS_QUARTER = [0.25] + HEAD_EPS
# the head shared by every encoder (Notes/REFACTOR_LOG.md, 26 Sept): lambda 0.3, gamma 0.05 as Adult, T = 150
# (the certificate is T-invariant under contraction; the trajectory settles within 20 steps at lr 2)
FINAL = dict(d=8, lam=0.3, gamma=0.05, lr=2.0, decay=0.01, T=150)

SST2 = dict(
    task="sst2",
    loader=load_task,
    encoder="frozen",
    adapter_dir=None,
    whiten=True,
    seed=0,
    eps=HEAD_EPS,
    delta=5e-6,
    seeds=41,
    tune_seeds=11,
    train_seeds=9,
    select_seeds=3,
    select_eps=[1, 2, 5],
    gamma_cauchy=2,
    k_grid=_RADII,
    bound_model="crown",
    stage1=dict(d=[8], lam=[0.3, 0.6], gamma=[0.01, 0.02, 0.05, 0.1, 0.2], lr=2.0, decay=0.01, T=50),
    stage2=dict(top=0, variants=[dict(T=300)]),
    final=dict(
        logistic=dict(FINAL), mlp=None
    ),  # the frozen arm uses the shared head too (its sweep is the record)
    arms=dict(mlp=dict(h=[8], d=[None], gamma=[0.02], lam=[0.05], T=[300])),
    long=None,
    warm=None,
    dpsgd_opt_approx=dict(q=[0.01, 0.05], steps=[300, 1000], lr=[0.5, 2.0], clip=[1.0]),
    dpsgd_opt_pure=dict(q=[0.05, 0.2, 1.0], steps=[5, 20, 100], lr=[2.0, 8.0, 15.0], gamma=[0.1]),
    pld_interval=1e-3,
    pate=None,  # the public set is the 872 dev sentences: no teacher pool
    groups=("logistic",),
    runs_dir="ags/sst2/runs/contractive_frozen",
    figures_dir="ags/sst2/figures",
    fig_stem="fig17_hybrid_sst2",
)


# the data-size panel's DP-SGD settings (batch, epochs), chosen on the public set at eps = 3: the full-data
# setting (1,024 x 3 epochs) leaves 2k and 10k sentences with too few optimizer steps (0.587 and 0.711 test
# against 0.778 and 0.863 with these)
PANEL_SETTING = {2000: "b128e10", 10000: "b512e6"}


def finetuned(eps1, n_private=None, final=FINAL):
    """The task on the encoder fine-tuned at eps1 (its adapter under runs/finetune/<tag>/adapter)."""
    tag = f"lora_eps{eps1:g}_n{n_private or 'all'}_s0" + (f"_{PANEL_SETTING[n_private]}" if n_private else "")
    name = f"sst2_ft_eps{eps1:g}" + (f"_n{n_private}" if n_private else "")
    return dict(
        SST2,
        task=name,
        encoder=tag,
        adapter_dir=f"ags/sst2/runs/finetune/{tag}/adapter",
        whiten=False,
        eps=HEAD_EPS_QUARTER if eps1 == 0.25 and not n_private else HEAD_EPS,
        n_private=n_private,
        stage1=dict(
            d=[final["d"]],
            lam=[final["lam"]],
            gamma=[final["gamma"]],
            lr=final["lr"],
            decay=final["decay"],
            T=final["T"],
        ),
        stage2=dict(top=0, variants=[]),
        final=dict(logistic=dict(final), mlp=None),
        runs_dir=f"ags/sst2/runs/contractive_{tag}",
    )


PANEL_N = [2000, 10000]  # the data-size panel at a total budget of 3: hybrids from eps_1 in PANEL_EPS1
PANEL_EPS1 = [0.5, 1, 2]


def frozen_n(n_private):
    """The frozen-encoder task on the first n_private private sentences (the pinned head, no sweep)."""
    return dict(
        SST2,
        task=f"sst2_n{n_private}",
        n_private=n_private,
        stage1=dict(
            d=[FINAL["d"]],
            lam=[FINAL["lam"]],
            gamma=[FINAL["gamma"]],
            lr=FINAL["lr"],
            decay=FINAL["decay"],
            T=FINAL["T"],
        ),
        stage2=dict(top=0, variants=[]),
        final=dict(logistic=dict(FINAL), mlp=None),
        runs_dir=f"ags/sst2/runs/contractive_frozen_n{n_private}",
    )


TASKS = {
    "sst2": SST2,
    **{f"sst2_ft_eps{e:g}": finetuned(e) for e in FINETUNE_EPS},
    **{f"sst2_n{n}": frozen_n(n) for n in PANEL_N},
    **{f"sst2_ft_eps{e:g}_n{n}": finetuned(e, n) for n in PANEL_N for e in PANEL_EPS1},
}
