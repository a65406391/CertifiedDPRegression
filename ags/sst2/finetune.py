"""DP-LoRA fine-tuning of all-mpnet-base-v2 on SST-2: the DP-SGD phase of the hybrid, and the end-to-end arm.

The recipe of Yu et al. (2022): LoRA (rank 16, alpha 32) on the attention q and v projections of every
layer plus a linear head on the mean-pooled output, trained with DP-SGD (Opacus: Poisson sampling at
expected batch `--batch`, per-example clipping at `--clip`, the PRV accountant) to (eps, delta). The
private set is ags.sst2.shared.split's private part (capped by --n-private); the public dev set and the
held-out test set are scored with the model's own head, which is the end-to-end DP fine-tuning arm.
Afterwards every train and dev sentence is embedded with the adapted encoder (adapter merged) and cached
under the run's tag, so the AGS head can be trained on the adapted features by the contractive driver.
--eps 0 trains without privacy (the ceiling). Writes runs/finetune/<tag>/{adapter/, head.pt, results.json}.

    PYTHONPATH=. python -m ags.sst2.finetune --eps 3 --device cuda:1
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ags.sst2.shared import BASE, MAX_LEN, embeddings, log, raw, split

RUNS = "ags/sst2/runs/finetune"


class Classifier(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.encoder = enc
        self.head = torch.nn.Linear(768, 2)

    def features(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * mask).sum(1) / mask.sum(1), dim=1)

    def forward(self, input_ids, attention_mask):
        return self.head(self.features(input_ids, attention_mask))


def tokenize(tok, sentences):
    enc = tok(sentences, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def accuracy(model, ids, am, y, device, batch=512):
    model.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(y), batch):
            logits = model(ids[i : i + batch].to(device), am[i : i + batch].to(device))
            hits += int((logits.argmax(1).cpu() == y[i : i + batch]).sum())
    model.train()
    return hits / len(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eps", type=float, required=True, help="0 for the non-private run")
    ap.add_argument("--delta", type=float, default=5e-6, help="the encoder's half of the pipeline's 1e-5")
    ap.add_argument("--n-private", type=int, default=None, dest="n_private")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024, help="expected Poisson batch")
    ap.add_argument("--physical", type=int, default=64, help="physical batch (memory)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"lora_eps{a.eps:g}_n{a.n_private or 'all'}_s{a.seed}"
    out = f"{RUNS}/{tag}"
    if os.path.exists(f"{out}/results.json"):
        log(f"{out}/results.json exists; nothing to do")
        return
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    tr_s, tr_y, dv_s, dv_y = raw()
    te, priv = split(len(tr_s), a.seed)
    if a.n_private:
        priv = priv[: a.n_private]
    tok = AutoTokenizer.from_pretrained(BASE)
    ids, am = tokenize(tok, [tr_s[i] for i in priv])
    y = torch.as_tensor(tr_y[priv])
    ids_dv, am_dv = tokenize(tok, dv_s)
    ids_te, am_te = tokenize(tok, [tr_s[i] for i in te])
    y_dv, y_te = torch.as_tensor(dv_y), torch.as_tensor(tr_y[te])
    log(f"{tag}: private {len(priv)}, public {len(dv_s)}, test {len(te)}; eps={a.eps:g} delta={a.delta:g}")

    enc = get_peft_model(
        AutoModel.from_pretrained(BASE),
        LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=["q", "v"], bias="none"),
    )
    model = Classifier(enc).to(a.device)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"trainable parameters {sum(p.numel() for p in params)}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    loader = DataLoader(TensorDataset(ids, am, y), batch_size=a.batch, shuffle=True)
    engine = None
    if a.eps > 0:
        from opacus import PrivacyEngine

        engine = PrivacyEngine(accountant="prv")
        model, opt, loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=loader,
            epochs=a.epochs,
            target_epsilon=a.eps,
            target_delta=a.delta,
            max_grad_norm=a.clip,
            poisson_sampling=True,
        )
        log(f"noise multiplier {opt.noise_multiplier:.4f}, sample rate {loader.sample_rate:.5f}")
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    # logical optimizer steps: the memory manager splits each Poisson batch into physical ones
    steps = a.epochs * (round(1 / loader.sample_rate) if engine is not None else len(loader))
    model.train()
    for epoch in range(a.epochs):
        if engine is not None:
            from opacus.utils.batch_memory_manager import BatchMemoryManager

            ctx = BatchMemoryManager(data_loader=loader, max_physical_batch_size=a.physical, optimizer=opt)
        else:
            import contextlib

            ctx = contextlib.nullcontext(loader)
        with ctx as batches:
            for b_ids, b_am, b_y in batches:
                opt.zero_grad()
                loss = loss_fn(model(b_ids.to(a.device), b_am.to(a.device)), b_y.to(a.device))
                loss.backward()
                opt.step()
        acc_dv = accuracy(model, ids_dv, am_dv, y_dv, a.device)
        log(
            f"  epoch {epoch + 1}/{a.epochs}: {steps} optimizer steps, public acc {acc_dv:.4f}, {time.time() - t0:.0f}s"
        )
    res = dict(
        tag=tag,
        eps=a.eps,
        delta=a.delta,
        eps_spent=float(engine.get_epsilon(a.delta)) if engine is not None else 0.0,
        noise_multiplier=float(opt.noise_multiplier) if engine is not None else 0.0,
        n_private=int(len(priv)),
        epochs=a.epochs,
        batch=a.batch,
        lr=a.lr,
        rank=a.rank,
        clip=a.clip,
        seed=a.seed,
        steps=steps,
        select=accuracy(model, ids_dv, am_dv, y_dv, a.device),
        test=accuracy(model, ids_te, am_te, y_te, a.device),
        train_seconds=time.time() - t0,
    )
    core = model._module if hasattr(model, "_module") else model
    os.makedirs(out, exist_ok=True)
    core.encoder.save_pretrained(f"{out}/adapter")
    torch.save(core.head.state_dict(), f"{out}/head.pt")
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)
    log(
        f"{tag}: end-to-end public {res['select']:.4f} test {res['test']:.4f} (eps spent {res['eps_spent']:.3f})"
    )
    del model, opt, engine
    torch.cuda.empty_cache()
    embeddings(tag, f"{out}/adapter", a.device)


if __name__ == "__main__":
    main()
