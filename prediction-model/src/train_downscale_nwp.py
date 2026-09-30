"""
NWP-informed downscale: the local network corrects a global model instead of
guessing the atmosphere from one station.

DESIGN — why this is different from the previous attempt
---------------------------------------------------------
The earlier local-only and spatial-context models both started from
PERSISTENCE, so they could never beat a good NWP: they were learning a signal
we already knew was weaker. Here the network predicts the RESIDUAL of a
bias-corrected NWP forecast:

    station_value(t0+h)  ~=  a*NWP(t0+h) + b  +  network(local, NWP)  [delta]

The residual head is ZERO-INITIALISED, so at epoch 0 the model output is exactly
the calibrated global forecast. Training can only move it away from there if the
local telemetry genuinely adds information. The benchmark is therefore "NWP plus
a learned local correction", which is the standard statistical downscaling setup
and is the only framing in which parity with NWP is attainable.

FEATURES
--------
  own   : the station's 24h x 8 observation window (unchanged from the canonical set)
  nwp_t : NWP at the station's location at t0, t0-3h, t0-6h  (5 vars x 3 lags)
  nwp_h : NWP at t0+h (the forecast being corrected)
  off   : nwp_t - own(t0)   the systematic station/grid offset, the main learnable term
  diurnal: local hour-of-day sin/cos, so the offset can be allowed to vary by time of day

PROTOCOL
--------
  TRAIN  -> fit the affine NWP calibration (a, b)
  VAL    -> early stopping and checkpoint selection on per-variable MAE
  TEST   -> never read here; scored afterwards by benchmark_vs_nwp.py
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

from dataset import TelemetryDataPipeline, build_forecast_windows, DEFAULT_SEQ_LEN  # noqa: E402
from model import CfCCell  # noqa: E402

DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(DATA_DIR, "nwp_benchmark_cache.json")
HORIZON = int(os.environ.get("DOWN_HORIZON", "6"))
EPOCHS = int(os.environ.get("DOWN_EPOCHS", "80"))
PATIENCE = int(os.environ.get("DOWN_PATIENCE", "12"))
LR = float(os.environ.get("DOWN_LR", "1e-3"))
HIDDEN = int(os.environ.get("DOWN_HIDDEN", "64"))
SEED = 42

# model key in the NWP cache, telemetry column, units-normalised later
NWP_VARS = [
    ("temperature_2m", 0),      # temperature
    ("relative_humidity_2m", 2), # humidity
    ("surface_pressure", 3),     # pressure
    ("wind_speed_10m", 4),       # wind_speed
    ("precipitation", 7),        # precipitation
]
NWP_MODELS = ["ecmwf_ifs025", "gfs_seamless"]


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_nwp():
    with open(CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    best = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        base = f"{v['station_id']}|{v['model']}"
        if base not in best or len(v.get("time", [])) > best[base][1]:
            best[base] = (v, len(v.get("time", [])))
    return {b: {"idx": {t: i for i, t in enumerate(x[0]["time"])}, "rec": x[0]}
            for b, x in best.items()}


def ts_key(iso):
    return iso[:13] + ":00"


def minus_hours(iso, h):
    return (np.datetime64(iso[:19]) - np.timedelta64(h, "h")).astype(str)[:13] + ":00"


class DownscaleCfc(nn.Module):
    """CfC over the local window, fused with NWP state. Predicts an NWP residual."""

    def __init__(self, local_dim=8, nwp_dim=22, hidden=HIDDEN):
        super().__init__()
        self.local_enc = nn.Sequential(
            nn.Linear(local_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cfc = CfCCell(hidden, hidden)
        self.nwp_enc = nn.Sequential(
            nn.Linear(nwp_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.heads = nn.ModuleDict({
            v: nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for v in ("temperature", "humidity", "pressure", "wind_speed", "precipitation")
        })
        # ZERO-INITIALISED residual head: the model starts as the calibrated NWP.
        for v in self.heads:
            nn.init.zeros_(self.heads[v][-1].weight)
            nn.init.zeros_(self.heads[v][-1].bias)

    def forward(self, x, dt, nwp):
        h = self.local_enc(x[:, 0])
        for t in range(x.shape[1]):
            h = self.cfc(self.local_enc(x[:, t]), h, dt[:, t])
        fused = self.fuse(torch.cat([h, self.nwp_enc(nwp)], dim=-1))
        return {v: self.heads[v](fused).squeeze(-1) for v in self.heads}


def clamp_var(v, x):
    if v == "temperature":
        return min(60.0, max(-20.0, x))
    if v == "humidity":
        return min(100.0, max(0.0, x))
    if v == "pressure":
        return min(1100.0, max(850.0, x))
    if v == "wind_speed":
        return max(0.0, x)
    if v == "precipitation":
        return max(0.0, x)
    return x


TARGETS = ["temperature", "humidity", "pressure", "wind_speed"]
SCALAR_TARGETS = TARGETS + ["precipitation"]

# dataset metadata uses `actual_precip_mm` for precipitation, not `target_*`.
# Map each variable to its metadata key so we never silently read a missing one.
META_KEY = {v: f"target_{v}" for v in TARGETS}
META_KEY["precipitation"] = "actual_precip_mm"


def meta_val(m, var):
    return m[META_KEY[var]]


def meta_origin(m, var):
    return m[f"origin_{var}"]


# column index inside the NWP feature block (5 vars x 4 slots)
NWP_COL = {"temperature": 0, "humidity": 1, "pressure": 2, "wind_speed": 3,
           "precipitation": 4}


def build_split(pipe, nwp, split, h, model_key):
    res = build_forecast_windows(pipeline=pipe, split=split, horizon=h,
                                 seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    X = res[0].detach().cpu().numpy().astype(np.float64)
    dt = res[1].detach().cpu().numpy().astype(np.float32)
    meta = res[6]
    n = len(X)
    keep = []
    nwp_rows = np.full((n, 22), np.nan, dtype=np.float64)
    for i, m in enumerate(meta):
        e = nwp.get(f"{m['station_id']}|{model_key}")
        if not e:
            continue
        k0, kh = ts_key(m["origin_timestamp"]), ts_key(m["target_timestamp"])
        k3, k6 = minus_hours(m["origin_timestamp"], 3), minus_hours(m["origin_timestamp"], 6)
        idx = e["idx"]
        if not all(k in idx for k in (k0, kh, k3, k6)):
            continue
        vals = []
        for field, _ in NWP_VARS:
            a0 = e["rec"][field][idx[k0]]
            ah = e["rec"][field][idx[kh]]
            a3 = e["rec"][field][idx[k3]]
            a6 = e["rec"][field][idx[k6]]
            if None in (a0, ah, a3, a6):
                vals = None
                break
            vals += [a0, a0 - a3, 0.0, ah]
        if vals is None:
            continue
        # layout: 5 vars x [t0, 3h-tendency, (reserved), t0+h] = 20, + 2 diurnal
        hour = int(np.datetime64(m["origin_timestamp"][:13]).astype(int)) % 24
        vals += [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)]
        nwp_rows[i] = vals
        keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    return X[keep], dt[keep], nwp_rows[keep], [meta[i] for i in keep]


def main():
    set_seed(SEED)
    pipe = TelemetryDataPipeline()
    nwp = load_nwp()
    H = HORIZON

    print("=" * 100)
    print(f"NWP-INFORMED DOWNSCALE — horizon +{H}h, model {os.environ.get('DOWN_MODEL','ecmwf_ifs025')}")
    print("=" * 100)

    for model_key in [m for m in NWP_MODELS if os.environ.get("DOWN_MODEL") == m] or NWP_MODELS:
        print(f"\n{'#' * 100}\n# NWP source: {model_key}\n{'#' * 100}")
        trX, trdt, trn, trm = build_split(pipe, nwp, "train", H, model_key)
        vaX, vadt, van, vam = build_split(pipe, nwp, "val", H, model_key)
        print(f"  train {len(trm)} windows, val {len(vam)} windows")

        # ---- affine NWP calibration, fitted on TRAIN --------------------------
        # Only the t0+h slot (index 3 of each variable's block) is calibrated;
        # the t0 / tendency slots are context features and are not targets.
        calib = {}
        for v in SCALAR_TARGETS:
            ci = NWP_COL[v]
            pp = trn[:, ci * 4 + 3]   # NWP at t0+h
            pt = np.array([meta_val(m, v) for m in trm], float)
            ok = np.isfinite(pp) & np.isfinite(pt)
            if ok.sum() >= 200:
                a, b = np.polyfit(pp[ok], pt[ok], 1)
            else:
                a, b = 1.0, 0.0
            calib[v] = (float(a), float(b))
        print("  affine calibration (train): " + ", ".join(
            f"{v}={calib[v][0]:.3f}x{calib[v][1]:+.3f}" for v in SCALAR_TARGETS))

        def base_forecast(rows, var):
            a, b = calib[var]
            ci = NWP_COL[var]
            out = np.array([a * r[ci * 4 + 3] + b for r in rows], float)
            return np.array([clamp_var(var, v) for v in out])

        # ---- feature normalisation (train only) ------------------------------
        mu = np.nanmean(trn, axis=0)
        sd = np.nanstd(trn, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        trn_n = np.nan_to_num((trn - mu) / sd, nan=0.0)
        van_n = np.nan_to_num((van - mu) / sd, nan=0.0)

        mmu = pipe.norm_means.astype(np.float64)
        msd = pipe.norm_stds.astype(np.float64)
        trX_n = ((trX - mmu) / msd).astype(np.float32)
        vaX_n = ((vaX - mmu) / msd).astype(np.float32)

        tgt_tr = {v: np.array([meta_val(m, v) for m in trm], float) for v in SCALAR_TARGETS}
        tgt_va = {v: np.array([meta_val(m, v) for m in vam], float) for v in SCALAR_TARGETS}
        # Keep the calibrated-NWP base as torch tensors so the residual head's
        # gradient flows. Converting the base to numpy and adding would detach
        # the network output and train nothing.
        base_tr = {v: torch.tensor(base_forecast(trn, v), dtype=torch.float32)
                   for v in SCALAR_TARGETS}

        model = DownscaleCfc(nwp_dim=22)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  model: DownscaleCfc, {n_par} parameters")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        tX = torch.tensor(trX_n); tdt = torch.tensor(trdt); tn = torch.tensor(trn_n, dtype=torch.float32)
        vX = torch.tensor(vaX_n); vdt = torch.tensor(vadt); vn = torch.tensor(van_n, dtype=torch.float32)
        base_va = {v: base_forecast(van, v) for v in SCALAR_TARGETS}

        def evaluate():
            model.eval()
            out = {v: np.zeros(len(vam)) for v in SCALAR_TARGETS}
            with torch.no_grad():
                for i in range(0, len(vam), 512):
                    sl = slice(i, min(i + 512, len(vam)))
                    d = model(vX[sl], vdt[sl], vn[sl])
                    for v in SCALAR_TARGETS:
                        out[v][sl] = base_va[v][sl] + d[v].numpy()
            return out

        best = {"score": float("inf"), "epoch": -1, "state": None}
        patience = 0
        idx_all = np.arange(len(trm))
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(idx_all)
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                if len(b) < 8:
                    continue
                d = model(tX[b], tdt[b], tn[b])
                loss = 0.0
                for v in TARGETS:
                    pred = base_tr[v][b] + d[v]
                    loss = loss + nn.functional.smooth_l1_loss(
                        pred, torch.tensor(tgt_tr[v][b], dtype=torch.float32))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            out = evaluate()
            maes = np.array([np.abs(out[v] - tgt_va[v]).mean() for v in TARGETS])
            base_maes = np.array([np.abs(base_va[v] - tgt_va[v]).mean() for v in TARGETS])
            sel = float(maes.mean())
            if sel < best["score"] - 1e-5:
                best = {"score": sel, "epoch": ep,
                        "state": {k: v2.detach().clone() for k, v2 in model.state_dict().items()}}
                patience = 0
            else:
                patience += 1
            if ep % 10 == 0 or patience >= PATIENCE:
                print(f"  ep{ep:>3} val_mae={np.round(maes,3).tolist()} "
                      f"(NWP base {np.round(base_maes,3).tolist()})")
            if patience >= PATIENCE:
                print(f"  early stop ep{ep}, best ep{best['epoch']}")
                break

        model.load_state_dict(best["state"])
        out = evaluate()
        print(f"\n  VALIDATION +{H}h  [{model_key}]  best epoch {best['epoch']}")
        print(f"  {'variable':<14}{'persistence':>13}{'cal NWP':>10}{'NWP+down':>11}"
              f"{'gain vs NWP':>13}{'gain vs pers':>13}")
        res = {}
        for v in TARGETS:
            p_mae = np.abs(tgt_va[v] - np.array([meta_origin(m, v) for m in vam])).mean()
            n_mae = np.abs(base_va[v] - tgt_va[v]).mean()
            d_mae = np.abs(out[v] - tgt_va[v]).mean()
            g_nwp = (n_mae - d_mae) / n_mae * 100
            g_pers = (p_mae - d_mae) / p_mae * 100
            print(f"  {v:<14}{p_mae:>13.3f}{n_mae:>10.3f}{d_mae:>11.3f}"
                  f"{g_nwp:>+12.1f}%{g_pers:>+12.1f}%")
            res[v] = {"persistence": float(p_mae), "nwp": float(n_mae),
                      "nwp_downscale": float(d_mae),
                      "gain_vs_nwp_pct": float(g_nwp), "gain_vs_persistence_pct": float(g_pers)}
        outp = os.path.join(DATA_DIR, f"downscale_h{H}_{model_key}.json")
        with open(outp, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"horizon_h": H, "nwp_model": model_key, "parameters": n_par,
                       "best_epoch": best["epoch"], "validation": res}, f, indent=2)
        print(f"  written: {outp}")


if __name__ == "__main__":
    main()
