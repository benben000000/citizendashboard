"""
Bounded feasibility experiment: can a spatially-aware, context-enriched CfC
close the gap to ECMWF?

This is a deliberately SMALL, well-controlled experiment, not a full production
retrain. Its purpose is to answer one question with evidence:

    Does adding (a) the already-built 75-feature station context and
    (b) nearest-neighbour station values at the forecast origin
    let the network beat persistence by enough to be competitive with a global
    NWP model?

If the answer is no, no amount of further tuning will fix it, because the
information simply is not in local station telemetry.

PROTOCOL (differs deliberately from the existing trainer)
---------------------------------------------------------
* Checkpoint selection on PER-VARIABLE validation MAE, not the summed training
  loss. The existing trainer selected on total validation loss, which is
  dominated by the temperature term, so the rain and water heads were effectively
  selected on temperature performance.
* Real early stopping with patience on the same per-variable criterion.
* Adequate epochs (the existing candidate run used epochs=5 on an 11k-parameter
  network over ~3.5k windows).
* TRAIN for fitting, VALIDATION for selection and early stopping.
  TEST is never read here.

Outputs a report to data/spatial_cfc_feasibility.json.
"""

import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dataset import (  # noqa: E402
    TelemetryDataPipeline,
    build_forecast_windows,
    build_feature_augmented_forecast_windows,
    extract_zero_leakage_feature_vector,
    normalize_features,
    DEFAULT_SEQ_LEN,
    FEATURE_AUGMENTED_SCHEMA,
)
from model import CfCCell  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
HORIZON = int(os.environ.get("TRAIN_HORIZON", "6"))
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "60"))
PATIENCE = int(os.environ.get("TRAIN_PATIENCE", "10"))
LR = float(os.environ.get("TRAIN_LR", "2e-3"))
BATCH = 64
K_NBRS = 3
SEED = 42

TARGETS = ["temperature", "humidity", "pressure", "wind_speed"]
TARGET_COL = {"temperature": 0, "humidity": 2, "pressure": 3, "wind_speed": 4}


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def haversine(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(o2 - o1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


class SpatialCfc(nn.Module):
    """
    CfC over the station's own 24h sequence, fused with an encoder over
    [own 75-feature context | K neighbours' current 8-feature observation].

    Heads predict origin-anchored deltas and are zero-initialised, so the model
    STARTS as exact persistence and must learn only the correction. That makes
    it impossible for training to do worse than persistence at initialisation.
    """

    def __init__(self, input_dim=8, context_dim=75, nbr_dim=24, hidden=32):
        super().__init__()
        self.seq_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cfc = CfCCell(hidden, hidden)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.nbr_encoder = nn.Sequential(
            nn.Linear(nbr_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.heads = nn.ModuleDict({
            v: nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for v in TARGETS
        })
        self.rain_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        for v in TARGETS:
            nn.init.zeros_(self.heads[v][-1].weight)
            nn.init.zeros_(self.heads[v][-1].bias)

    def forward(self, x, dt, ctx, nbr, origin):
        h = self.seq_encoder(x[:, 0])
        for t in range(x.shape[1]):
            h = self.cfc(self.seq_encoder(x[:, t]), h, dt[:, t])
        fused = self.fusion(torch.cat(
            [h, self.ctx_encoder(ctx), self.nbr_encoder(nbr)], dim=-1))
        deltas = {v: self.heads[v](fused).squeeze(-1) for v in TARGETS}
        return deltas, self.rain_head(fused).squeeze(-1)


def build_nbr_matrix(pipe, meta, nbr_ids, n):
    """K neighbours' 8-feature observation at the forecast origin, plus validity mask."""
    out = np.zeros((n, K_NBRS * 8), dtype=np.float32)
    mask = np.zeros((n, K_NBRS), dtype=np.float32)
    for i, m in enumerate(meta):
        t0 = m["origin_timestamp"]
        for k, o in enumerate(nbr_ids):
            rec = pipe.station_hourly.get(o, {}).get(t0)
            if rec is None:
                continue
            out[i, k * 8:(k + 1) * 8] = [
                rec["temperature"], rec["heat_index"], rec["humidity"], rec["pressure"],
                rec["wind_speed"], rec["wind_sin"], rec["wind_cos"], rec["precipitation"]]
            mask[i, k] = 1.0
    return np.concatenate([out, mask], axis=1)


def main():
    set_seed(SEED)
    pipe = TelemetryDataPipeline()
    coords = json.load(open(os.path.join(DATA_DIR, "station_coords.json"), encoding="utf-8"))
    stations = sorted(pipe.station_hourly)
    lat = {s: float(coords[s]["lat"]) for s in stations}
    lon = {s: float(coords[s]["lon"]) for s in stations}
    nbrs = {s: [o for _, o in sorted(
        ((haversine(lat[s], lon[s], lat[o], lon[o]), o) for o in stations if o != s)
    )[:K_NBRS]] for s in stations}

    H = HORIZON
    print("=" * 96)
    print(f"SPATIAL-CFC FEASIBILITY — horizon +{H}h, epochs<={EPOCHS}, patience {PATIENCE}")
    print("=" * 96)

    cache = {}
    for split in ("train", "val"):
        # build_feature_augmented_forecast_windows returns:
        #   0 telemetry [N,24,8] | 1 context [N,75] | 2 dt [N,24,1]
        #   3 rain | 4 precip | 5 water | 6 has_water | 7 metadata
        res = build_feature_augmented_forecast_windows(
            pipeline=pipe, split=split, horizon=H, seq_len=DEFAULT_SEQ_LEN,
            return_metadata=True)
        X = res[0].detach().cpu().numpy().astype(np.float32)
        ctx = res[1].detach().cpu().numpy().astype(np.float32)
        dt = res[2].detach().cpu().numpy().astype(np.float32)
        rain = res[3].detach().cpu().numpy().astype(np.float32)
        precip = res[4].detach().cpu().numpy().astype(np.float32)
        meta = res[7]
        n = len(X)
        nbr = np.zeros((n, K_NBRS * 8), dtype=np.float32)
        mask = np.zeros((n, K_NBRS), dtype=np.float32)
        for i, m in enumerate(meta):
            t0 = m["origin_timestamp"]
            for k, o in enumerate(nbrs[m["station_id"]]):
                rec = pipe.station_hourly.get(o, {}).get(t0)
                if rec is None:
                    continue
                nbr[i, k * 8:(k + 1) * 8] = [
                    rec["temperature"], rec["heat_index"], rec["humidity"],
                    rec["pressure"], rec["wind_speed"], rec["wind_sin"],
                    rec["wind_cos"], rec["precipitation"]]
                mask[i, k] = 1.0
        nbr = np.concatenate([nbr, mask], axis=1)
        cache[split] = dict(
            x=torch.tensor(X), ctx=torch.tensor(ctx), dt=torch.tensor(dt),
            rain=torch.tensor(rain).squeeze(-1), precip=torch.tensor(precip).squeeze(-1),
            nbr=torch.tensor(nbr), meta=meta, n=n)
        print(f"  {split}: {n} windows")

    tr, va = cache["train"], cache["val"]
    nbr_dim = K_NBRS * 8 + K_NBRS
    # neighbour features use the same 8-feature scale as the station window
    mean_nb = tr["nbr"][:, :K_NBRS * 8].reshape(-1, 8).mean(0)
    std_nb = tr["nbr"][:, :K_NBRS * 8].reshape(-1, 8).std(0).clamp_min(1e-3)
    tr_nb = (tr["nbr"] - torch.cat([mean_nb.repeat(K_NBRS), torch.zeros(K_NBRS)])) / \
            torch.cat([std_nb.repeat(K_NBRS), torch.ones(K_NBRS)])
    va_nb = (va["nbr"] - torch.cat([mean_nb.repeat(K_NBRS), torch.zeros(K_NBRS)])) / \
            torch.cat([std_nb.repeat(K_NBRS), torch.ones(K_NBRS)])

    def origin_weather(split):
        d = cache[split]
        return np.array([[m["origin_temperature"], m["origin_humidity"],
                          m["origin_pressure"], m["origin_wind_speed"]]
                         for m in d["meta"]], dtype=np.float32)

    def target_weather(split):
        d = cache[split]
        return np.array([[m[f"target_{v}"] for v in TARGETS]
                         for m in d["meta"]], dtype=np.float32)

    tr_o, va_o = origin_weather("train"), origin_weather("val")
    tr_t, va_t = target_weather("train"), target_weather("val")

    model = SpatialCfc(context_dim=tr["ctx"].shape[1], nbr_dim=nbr_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: SpatialCfc, {n_params} parameters, context {tr['ctx'].shape[1]}, nbr {nbr_dim}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    idx_all = np.arange(tr["n"])

    def evaluate(split, nb):
        d = cache[split]
        model.eval()
        preds = np.zeros((d["n"], len(TARGETS)), dtype=np.float32)
        rainp = np.zeros(d["n"], dtype=np.float32)
        with torch.no_grad():
            for i in range(0, d["n"], 256):
                sl = slice(i, min(i + 256, d["n"]))
                deltas, rl = model(d["x"][sl], d["dt"][sl], d["ctx"][sl], nb[sl],
                                   torch.tensor(origin_weather(split)[sl]))
                for k, v in enumerate(TARGETS):
                    preds[sl, k] = origin_weather(split)[sl, k] + deltas[v].numpy()
                rainp[sl] = torch.sigmoid(rl).numpy()
        return preds, rainp

    best = {"score": float("inf"), "epoch": -1, "state": None}
    patience = 0
    history = []

    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(idx_all)
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 8:
                continue
            bt = torch.tensor(tr_o[b])
            deltas, rl = model(tr["x"][b], tr["dt"][b], tr["ctx"][b], tr_nb[b], bt)
            loss = 0.0
            for k, v in enumerate(TARGETS):
                loss = loss + nn.functional.smooth_l1_loss(deltas[v], torch.tensor(
                    tr_t[b, k] - tr_o[b, k]))
            y_rain = (tr["precip"][b] > 0.1).float()
            loss = loss + 0.5 * nn.functional.binary_cross_entropy_with_logits(rl, y_rain)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        sched.step()

        # PER-VARIABLE validation MAE as the selection criterion.
        vp, _ = evaluate("val", va_nb)
        maes = np.abs(vp - va_t).mean(axis=0)
        pers = np.abs(va_t - va_o).mean(axis=0)
        skill = (pers - maes) / pers * 100.0
        sel = float(maes.mean())
        history.append({"epoch": ep, "train_loss": tot / max(1, len(perm) // BATCH),
                        "val_mae_per_var": maes.tolist(), "skill_pct": skill.tolist()})
        if sel < best["score"] - 1e-5:
            best = {"score": sel, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "maes": maes.tolist(), "skill": skill.tolist()}
            patience = 0
        else:
            patience += 1
        if ep % 5 == 0 or patience >= PATIENCE:
            print(f"  ep{ep:>3} train={tot / max(1, len(perm) // BATCH):.4f} "
                  f"val_mae={np.round(maes, 3).tolist()} skill={np.round(skill, 1).tolist()}")
        if patience >= PATIENCE:
            print(f"  early stop at epoch {ep} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    vp, vr = evaluate("val", va_nb)
    vpersist = np.abs(va_t - va_o).mean(axis=0)
    vmae = np.abs(vp - va_t).mean(axis=0)
    vskill = (vpersist - vmae) / vpersist * 100.0

    # rain: Brier on physical precipitation definition
    y_rain = (va["precip"].numpy() > 0.1).astype(float)
    brier = float(np.mean((vr - y_rain) ** 2))
    last = np.where(va["precip"].numpy() <= 0.0, 0.0, 0.85)
    p_last = np.where(va["precip"].numpy() > 0.0, 0.85, 0.05)
    brier_p = float(np.mean((p_last - y_rain) ** 2))

    print(f"\n  VALIDATION, +{H}h, best epoch {best['epoch']} of {EPOCHS}")
    print(f"  {'variable':<14}{'persistence':>13}{'model':>10}{'skill':>9}")
    for k, v in enumerate(TARGETS):
        print(f"  {v:<14}{vpersist[k]:>13.3f}{vmae[k]:>10.3f}{vskill[k]:>+8.1f}%")
    print(f"  {'rain Brier':<14}{brier_p:>13.4f}{brier:>10.4f}"
          f"{(brier_p - brier) / brier_p * 100:>+8.1f}%")

    report = {
        "horizon_h": H, "epochs_run": len(history), "best_epoch": best["epoch"],
        "parameters": n_params, "context_dim": int(tr["ctx"].shape[1]), "nbr_dim": nbr_dim,
        "validation": {v: {"persistence": float(vpersist[k]), "model": float(vmae[k]),
                           "skill_pct": float(vskill[k])} for k, v in enumerate(TARGETS)},
        "validation_rain_brier": {"model": brier, "persistence": brier_p},
        "history": history,
    }
    out = os.path.join(DATA_DIR, f"spatial_cfc_feasibility_h{H}.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
