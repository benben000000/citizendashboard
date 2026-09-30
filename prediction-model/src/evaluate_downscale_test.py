"""
Final honest evaluation of the NWP-downscale path, on the TEST split.

TRAIN -> fit the affine coefficients
VAL   -> choose lambda (shrinkage toward identity) and the pooling level
TEST  -> report, ONCE, and never feed anything back

This separation matters because the previous run selected lambda on validation
and I have not yet checked whether the gain survives on data that played no part
in selection. Validation-selected gains routinely shrink or invert on test, and
reporting the validation number as if it were the outcome would be dishonest.

Reported per variable, at every horizon:
    persistence        last observation held forward
    raw NWP            the global model, unmodified
    calibrated NWP     affine fit, shrunk toward identity by the chosen lambda
    NWP + downscale    calibrated NWP plus the network's learned residual

and the previously-measured LNN test MAE for the same variable and horizon, so
the comparison is against what actually ships today.
"""

import json
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
MODELS = ["ecmwf_ifs025", "gfs_seamless"]
EPOCHS = int(os.environ.get("FIN_EPOCHS", "40"))
PATIENCE = int(os.environ.get("FIN_PATIENCE", "8"))
LR = float(os.environ.get("FIN_LR", "1e-3"))
HIDDEN = 64
SEED = 42

# name -> (nwp field, telemetry key, unit scale into STATION units, clamp)
VARS = {
    "temperature": ("temperature_2m", "target_temperature", 1.0, (-20.0, 60.0)),
    "humidity": ("relative_humidity_2m", "target_humidity", 1.0, (0.0, 100.0)),
    "pressure": ("surface_pressure", "target_pressure", 1.0, (850.0, 1100.0)),
    "wind_speed": ("wind_speed_10m", "target_wind_speed", 1.0 / 3.6, (0.0, None)),
}
# The network only ever corrects the four continuous heads. Rain is a
# classification problem and is benchmarked separately by benchmark_vs_nwp.py;
# folding it in here would let a regression residual distort the Brier score.
CONT = list(VARS.keys())


def clamp(v, x):
    lo, hi = VARS[v][3]
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_nwp():
    with open(CACHE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    best = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        b = f"{v['station_id']}|{v['model']}"
        if b not in best or len(v.get("time", [])) > len(best[b]["time"]):
            best[b] = v
    return best


def collect(pipe, best, split, h, model):
    res = build_forecast_windows(pipeline=pipe, split=split, horizon=h,
                                 seq_len=DEFAULT_SEQ_LEN, return_metadata=True)
    X = res[0].detach().cpu().numpy().astype(np.float64)
    dt = res[1].detach().cpu().numpy().astype(np.float32)
    meta = res[6]
    keep, feats = [], []
    rej = {"no_record": 0, "missing_key": 0, "null_field": 0}
    for i, m in enumerate(meta):
        rec = best.get(f"{m['station_id']}|{model}")
        if rec is None:
            rej["no_record"] += 1
            continue
        ix = rec.get("_ix")
        if ix is None:
            ix = {t: j for j, t in enumerate(rec["time"])}
            rec["_ix"] = ix
        t0 = m["origin_timestamp"][:13] + ":00"
        th = m["target_timestamp"][:13] + ":00"
        k3 = _minus(m["origin_timestamp"], 3)
        k6 = _minus(m["origin_timestamp"], 6)
        if not all(k in ix for k in (t0, th, k3, k6)):
            rej["missing_key"] += 1
            continue
        row = []
        for v, (fld, _mk, sc, _c) in VARS.items():
            a0 = rec[fld][ix[t0]]; ah = rec[fld][ix[th]]
            a3 = rec[fld][ix[k3]]; a6 = rec[fld][ix[k6]]
            if None in (a0, ah, a3, a6):
                row = None
                break
            a0, ah, a3, a6 = a0 * sc, ah * sc, a3 * sc, a6 * sc
            row += [a0, a0 - a3, 0.0, ah]
        if row is None:
            rej["null_field"] += 1
        if row is None:
            continue
        hour = int(m["origin_timestamp"][11:13])
        row += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
        feats.append(row)
        keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    if os.environ.get("FIN_DEBUG"):
        print(f"    [collect] split={split} h={h} model={model} meta={len(meta)} "
              f"kept={len(keep)} rejected={rej}")
    return X[keep], dt[keep], np.array(feats, float), [meta[i] for i in keep]


def _minus(iso, h):
    return (np.datetime64(iso[:19]) - np.timedelta64(h, "h")).astype(str)[:13] + ":00"


def fit(x, y, lam):
    if len(x) < 100:
        return 1.0, 0.0
    a, b = np.polyfit(x, y, 1)
    return float(1.0 + lam * (a - 1.0)), float(lam * b)


class DownscaleCfc(nn.Module):
    def __init__(self, nwp_dim, hidden=HIDDEN):
        super().__init__()
        self.loc = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cfc = CfCCell(hidden, hidden)
        self.nwp = nn.Sequential(nn.Linear(nwp_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))
        self.fuse = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))
        self.heads = nn.ModuleDict({
            v: nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for v in CONT})
        for v in CONT:
            nn.init.zeros_(self.heads[v][-1].weight)
            nn.init.zeros_(self.heads[v][-1].bias)

    def forward(self, x, dt, nwp):
        h = self.loc(x[:, 0])
        for t in range(x.shape[1]):
            h = self.cfc(self.loc(x[:, t]), h, dt[:, t])
        f = self.fuse(torch.cat([h, self.nwp(nwp)], dim=-1))
        return {v: self.heads[v](f).squeeze(-1) for v in CONT}


def main():
    set_seed(SEED)
    pipe = TelemetryDataPipeline()
    best = load_nwp()
    horizons = [int(x) for x in os.environ.get("FIN_HORIZONS", "1,6,12,24").split(",")]

    report = {"protocol": "fit=TRAIN, select lambda/mode=VAL, report=TEST (single pass)",
              "results": {}}

    for h in horizons:
        print("=" * 104)
        print(f"TEST-SPLIT EVALUATION  +{h}h")
        print("=" * 104)
        trX, trdt, trF, trm = collect(pipe, best, "train", h, "ecmwf_ifs025")
        vaX, vadt, vaF, vam = collect(pipe, best, "val", h, "ecmwf_ifs025")
        teX, tedt, teF, tem = collect(pipe, best, "test", h, "ecmwf_ifs025")
        print(f"  windows  train {len(trm)}   val {len(vam)}   test {len(tem)}\n")

        # ---- TRAIN: fit affine, VAR: pick lambda + pooling on validation ------
        chosen = {}
        for v, (_f, mk, sc, _c) in VARS.items():
            xtr = trF[:, CONT.index(v) * 4 + 3]
            ytr = np.array([m[mk] for m in trm], float)
            xva = vaF[:, CONT.index(v) * 4 + 3]
            yva = np.array([m[mk] for m in vam], float)
            s_tr = np.array([m["station_id"] for m in trm])
            s_va = np.array([m["station_id"] for m in vam])
            cands = {}
            for mode in ("pooled", "per_station"):
                for lam in np.arange(0.0, 1.01, 0.1):
                    p = np.zeros_like(yva)
                    if mode == "pooled":
                        a, b = fit(xtr, ytr, lam)
                        p = np.clip(a * xva + b, VARS[v][3][0], VARS[v][3][1])
                    else:
                        for s in np.unique(s_va):
                            mv, mt = s_va == s, s_tr == s
                            a, b = fit(xtr[mt], ytr[mt], lam)
                            p[mv] = np.clip(a * xva[mv] + b, VARS[v][3][0], VARS[v][3][1])
                    cands[(mode, round(float(lam), 2))] = float(np.abs(p - yva).mean())
            bk = min(cands, key=cands.get)
            chosen[v] = {"mode": bk[0], "lambda": bk[1]}

        # ---- apply the chosen scheme: coefficients come from TRAIN, applied
        #      unchanged to any split. Fitting on the split being scored would
        #      leak, and refitting per split is what the previous version did.
        def fit_coeffs():
            c = {}
            for v, (_f, mk, _sc, _b) in VARS.items():
                x = trF[:, CONT.index(v) * 4 + 3]
                y = np.array([m[mk] for m in trm], float)
                s = np.array([m["station_id"] for m in trm])
                lam = chosen[v]["lambda"]
                if chosen[v]["mode"] == "pooled":
                    c[v] = {"global": fit(x, y, lam), "per": {}}
                else:
                    per = {}
                    for st in np.unique(s):
                        mt = s == st
                        per[st] = fit(x[mt], y[mt], lam)
                    c[v] = {"global": (1.0, 0.0), "per": per}
            return c

        COEF = fit_coeffs()

        def calibrated(rows, metas):
            out = {}
            s_all = np.array([m["station_id"] for m in metas])
            for v, (_f, mk, _sc, (lo, hi)) in VARS.items():
                x = rows[:, CONT.index(v) * 4 + 3]
                cc = COEF[v]
                p = np.zeros(len(x), float)
                if cc["per"]:
                    for st in np.unique(s_all):
                        mv = s_all == st
                        a, b = cc["per"].get(st, cc["global"])
                        p[mv] = a * x[mv] + b
                else:
                    a, b = cc["global"]
                    p = a * x + b
                out[v] = np.clip(p, lo, hi)
            return out

        cal_te = calibrated(teF, tem)

        # ---- normalise features on TRAIN only --------------------------------
        mu, sd = np.nanmean(trF, 0), np.nanstd(trF, 0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        mmu = pipe.norm_means.astype(np.float64)
        msd = pipe.norm_stds.astype(np.float64)

        tX = torch.tensor(((trX - mmu) / msd).astype(np.float32))
        vX = torch.tensor(((vaX - mmu) / msd).astype(np.float32))
        eX = torch.tensor(((teX - mmu) / msd).astype(np.float32))
        tF_ = torch.tensor(np.nan_to_num((trF - mu) / sd, nan=0.0), dtype=torch.float32)
        vF_ = torch.tensor(np.nan_to_num((vaF - mu) / sd, nan=0.0), dtype=torch.float32)
        eF_ = torch.tensor(np.nan_to_num((teF - mu) / sd, nan=0.0), dtype=torch.float32)
        tdt = torch.tensor(trdt); vdt = torch.tensor(vadt); edt = torch.tensor(tedt)

        cal_tr = {v: torch.tensor(calibrated(trF, trm)[v], dtype=torch.float32)
                  for v in CONT}
        cal_va = calibrated(vaF, vam)
        cal_te_t = {v: torch.tensor(cal_te[v], dtype=torch.float32) for v in CONT}
        tgt_va = {v: np.array([m[VARS[v][1]] for m in vam], float) for v in CONT}
        tgt_tr = {v: np.array([m[VARS[v][1]] for m in trm], float) for v in CONT}

        model = DownscaleCfc(nwp_dim=trF.shape[1])
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        def predict(split):
            if split == "val":
                X, D, F_, base = vX, vdt, vF_, cal_va
            elif split == "train":
                X, D, F_, base = tX, tdt, tF_, {v: cal_tr[v].numpy() for v in CONT}
            else:
                X, D, F_, base = eX, edt, eF_, {v: cal_te[v] for v in CONT}
            model.eval()
            out = {v: np.array(base[v], dtype=float, copy=True) for v in CONT}
            with torch.no_grad():
                for i in range(0, len(X), 512):
                    sl = slice(i, min(i + 512, len(X)))
                    d = model(X[sl], D[sl], F_[sl])
                    for v in CONT:
                        out[v][sl] = np.clip(
                            out[v][sl] + d[v].numpy(), VARS[v][3][0], VARS[v][3][1])
            return out

        # NOTE: this must NOT be called `best` -- that name is the NWP lookup
        # table, and shadowing it made every later horizon find zero records.
        ckpt = {"score": float("inf"), "epoch": -1, "state": None}
        pat = 0
        idx_all = np.arange(len(trm))
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(idx_all)
            for i in range(0, len(perm), 64):
                b = perm[i:i + 64]
                if len(b) < 8:
                    continue
                d = model(tX[b], tdt[b], tF_[b])
                loss = 0.0
                for v in CONT:
                    loss = loss + nn.functional.smooth_l1_loss(
                        cal_tr[v][b] + d[v],
                        torch.tensor(tgt_tr[v][b], dtype=torch.float32))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            pv = predict("val")
            score = float(np.mean([np.abs(pv[v] - tgt_va[v]).mean() for v in CONT]))
            if score < ckpt["score"] - 1e-5:
                ckpt = {"score": score, "epoch": ep,
                        "state": {k: v2.detach().clone() for k, v2 in model.state_dict().items()}}
                pat = 0
            else:
                pat += 1
            if pat >= PATIENCE:
                break
        model.load_state_dict(ckpt["state"])

        pt = predict("test")
        print(f"  network {n_par} params, best epoch {ckpt['epoch']} of {EPOCHS}\n")
        print(f"  {'variable':<12}{'persist':>9}{'rawNWP':>9}{'calNWP':>9}{'+downscale':>11}"
              f"{'gain/cal':>10}{'skill/pers':>12}")
        print("  " + "-" * 72)
        rows = {}
        for v in CONT:
            y = np.array([m[VARS[v][1]] for m in tem], float)
            o = np.array([m[f"origin_{v}"] for m in tem], float)
            pmae = np.abs(y - o).mean()
            rmae = np.abs(teF[:, CONT.index(v) * 4 + 3] - y).mean()
            cmae = np.abs(cal_te[v] - y).mean()
            dmae = np.abs(pt[v] - y).mean()
            g = (cmae - dmae) / cmae * 100
            sk = (pmae - dmae) / pmae * 100
            print(f"  {v:<12}{pmae:>9.3f}{rmae:>9.3f}{cmae:>9.3f}{dmae:>11.3f}"
                  f"{g:>9.1f}%{sk:>11.1f}%")
            rows[v] = {"persistence": float(pmae), "raw_nwp": float(rmae),
                       "cal_nwp": float(cmae), "cal_nwp_downscale": float(dmae),
                       "gain_vs_cal_pct": float(g), "skill_vs_pers_pct": float(sk),
                       "calibration": chosen[v]}
        report["results"][str(h)] = rows
        report["results"][str(h)]["_best_epoch"] = ckpt["epoch"]
        report["results"][str(h)]["_params"] = n_par
        print()

    outp = os.path.join(DATA_DIR, "downscale_test_evaluation.json")
    with open(outp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
    print(f"written: {outp}")


if __name__ == "__main__":
    main()
