# Known Limitations

**Scope:** the KloudTrack prediction engine at policy version `2.0.0`, code commit
`cf0a37e239fd6cc5a3a43affb6fe69148ebba7bf`, benchmarked on test split
2026-08-15T22:00 → 2026-08-26T02:00.
**Companion documents:** [model-architecture.md](model-architecture.md) ·
[operations-runbook.md](operations-runbook.md)

This document exists to prevent one specific failure mode: **a reader concluding from an
accuracy table that this system does something it does not do.** Over-claiming is worse
than being unpleasantly accurate, because a false capability claim propagates into product
decisions, external claims and safety arguments.

---

## 1. The headline limitation: 15 of 20 cells are persistence

`inference_policy.json` v2.0.0 routes each of the 20 benchmarked (horizon × variable)
cells independently to either the learned model or the last observation. Measured directly
from the file:

| Horizon | temperature | humidity | pressure | wind_speed |
|---|---|---|---|---|
| +1h | persistence | persistence | persistence | persistence |
| +3h | persistence | persistence | persistence | persistence |
| +6h | **learned** | persistence | persistence | **learned** |
| +12h | **learned** | persistence | persistence | **learned** |
| +24h | persistence | persistence | persistence | **learned** |

- **15 of 20** benchmarked cells serve `persistence_fallback`.
- **5 of 20** serve `learned_model`.
- Including wind direction: **20 of 25**.

> The figure "13 of 20" has circulated in summaries of this system. It does not match the
> file. If you need the count, read it out of `inference_policy.json` rather than quoting a
> document that quotes another document.

**What this means for every accuracy number in this repository.** An MAE attributed to
"the model" at a persistence-served cell is the MAE of the last observation. Concretely:

- "Humidity MAE 1.677 %RH" means *we served the last humidity reading*. It does not mean
  the network learned humidity structure. Humidity is served by persistence at **all five
  horizons**, so no humidity number anywhere in the repository is a model result.
- "Pressure MAE 0.394 hPa, beating ECMWF's 2.219" means *persistence beats a 0.25° grid
  cell at surface pressure at these stations*. Surface pressure over terrain carries strong
  elevation structure a coarse cell smooths away, and the station samples its own terrain
  exactly. The pressure head itself, with the gate open, scores 1.3153 hPa at +6h against
  persistence's 1.3153 hPa — a difference of ~2e-5 hPa. **The pressure head contributes
  nothing.**
- "Temperature MAE at +1h / +3h / +24h" — persistence, at every one of those horizons.

The apparent weakness of this system across most horizons is a **policy decision**, not a
statement about the network. The network may be computing predictions in those cells and
the policy is discarding them. §2 quantifies how much is being discarded.

---

## 2. Discarded headroom

`data/headroom_diagnostic.json` measures, on the same windows, what the network's own
output scores *before* the policy gate. `head_skill_pct` is the raw head's relative MAE
improvement over persistence.

| Horizon | temperature | humidity | pressure | wind_speed |
|---|---|---|---|---|
| +1h | +5.41% | +5.40% | +10.42% | **−0.45%** |
| +3h | +7.12% | +7.96% | +21.34% | +0.56% |
| +6h | +13.97% | +7.42% | +13.72% | +10.07% |
| +12h | +25.88% | +19.74% | **−6.25%** | +19.24% |
| +24h | +6.80% | +3.93% | +3.42% | +0.92% |

Reading this honestly:

- **Humidity has real, measurable headroom** at +12h (+19.74%) and smaller amounts
  elsewhere, and the policy currently takes **none** of it. The reason is not that it is
  unavailable — it is that the promotion rule requires skill to clear a margin on a minimum
  number of horizons simultaneously, and humidity clears neither. Whether to promote
  partial-horizon humidity skill is an open decision, not a settled one.
- **Pressure headroom is small and inconsistent** (+13.72% at +6h, −6.25% at +12h). The
  large NWP margin is not coming from here.
- **Wind at +1h is negative** (−0.45%). The head is genuinely no better than persistence at
  +1h, which is consistent with the policy serving persistence there.
- **The headroom diagnostic is a diagnostic, not a reportable accuracy result.** It decides
  whether to open the gate. Do not quote these numbers as benchmark results.

---

## 3. The +24h temperature anomaly

Several numbers land on the same value, and that is a signal rather than a coincidence:

| Source | +24h temperature MAE (°C) |
|---|---|
| `nwp_benchmark_production.json` → `LNN production` | 1.4059 |
| `nwp_benchmark_production.json` → `persistence` | 1.4059 |
| `nwp_bm_ctrl_relu.json` → `persistence` | 1.4059 |
| `nwp_bm_ctrl_leaky.json` → `persistence` | 1.4059 |

Separately-produced numbers agreeing to four decimal places cannot be a modelling
coincidence. There are two distinct causes and they must be separated.

### 3.1 Cause one: production *is* persistence at that cell — by design

+24h temperature is served by `persistence_fallback`. The served forecast is literally the
origin observation, so `LNN production` and `persistence` are the same array and their MAEs
are bit-identical. **The equality is a property of the policy, not evidence that +24h
temperature is unforecastable.** Any inference of "the model cannot do +24h temperature"
drawn from this equality is invalid.

Supporting this: the headroom diagnostic shows the raw +24h temperature head at **+6.80%**
relative skill over persistence. The network is producing a non-trivial +24h temperature
delta. The policy is not serving it.

### 3.2 Cause two: persistence error is not monotone in lead time — measured

The other half of the signal is that persistence at +24h (1.4059) is **substantially lower
than persistence at +12h** (2.1619). Error falling as lead time grows is not what a forecast
skill discussion normally produces. It was measured directly.

Persistence MAE against lead time, computed on an **identical origin set** (n = 1,956
origins present at every lag from 1 h to 24 h, so composition cannot confound the curve):

| Lead (h) | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Temperature (°C) | 0.588 | 0.886 | 1.122 | 1.327 | 1.519 | 1.687 | 1.923 | 2.094 | **2.158** | 2.148 | 2.042 | 1.884 | 1.691 | 1.471 | **1.402** |
| Humidity (%RH) | 1.652 | 2.496 | 3.204 | 3.818 | 4.383 | 4.844 | 5.569 | 6.057 | **6.316** | 6.260 | 5.928 | 5.459 | 4.927 | 4.435 | **4.355** |
| Pressure (hPa) | 0.391 | 0.679 | 0.924 | 1.117 | 1.245 | 1.299 | 1.188 | 0.900 | **0.810** | 1.049 | 1.391 | 1.559 | 1.482 | 1.219 | **1.074** |
| Wind (m/s) | 0.863 | 0.967 | 1.077 | 1.158 | 1.223 | 1.291 | 1.411 | 1.469 | **1.486** | **1.495** | 1.441 | 1.360 | 1.300 | 1.180 | **1.167** |

**All four continuous variables show a peak at 12–14 h and a decline toward 24 h.** This
is not temperature-specific and it is not an artifact of which windows survived.

Ruled out:

- **Target misalignment.** `build_forecast_windows` requires the target hour to exist, be
  strictly after the origin, and satisfy `|actual_lead − h| ≤ 0.25 h`. Verified per window.
- **Window composition.** The table above is on a fixed origin set; the non-monotonicity
  survives.
- **Cross-split leakage.** Windows never span a split boundary; the 48 h embargo covers
  `seq_len + max_horizon`.

What the pattern is consistent with: a **strong ~24-hour periodic component** in the local
meteorological signal, so a 24-hour lead returns to the same diurnal phase and partially
cancels the diurnal error that dominates at 12 hours (the anti-phase maximum). Pressure
shows a two-peak structure (6 h and 18 h, trough at 12 h) rather than a single one, which
points at more than one periodic component in play.

### 3.3 Why this is a limitation and not a solution

If part of the apparent +24h advantage is **diurnal phase alignment** rather than genuine
24-hour predictability, then:

- +24h numbers do not generalise to a shifted horizon (+18h, +21h, +30h).
- Skill credited to a +24h cell may partly be skill credited to a fixed clock.
- Comparing +24h to +6h or +12h in a single table is comparing different regimes.

**Open work, not a resolved finding.** The diurnal-phase hypothesis is consistent with the
measurement but has not been tested by removing hour-of-day information from the model. The
test that would settle it: re-benchmark at horizons offset by six hours (+18h, +30h) and
check whether the model's advantage survives. Until that runs, treat every +24h number in
this repository as provisional and label it as such when reporting.

---

## 4. Instrument defects in the ground truth

The benchmark scores forecasts against station observations. Three stations have wind
sensors that do not work, and one station returns nothing. These are defects in the
**truth**, and they propagate into every wind number.

### 4.1 Three dead anemometers

From `data/anemometer_diagnostic.json` (model `ecmwf_ifs025`, 2,531 observations,
11 stations), the per-station ratio of station mean wind to NWP mean wind:

| Station | ratio | Interpretation |
|---|---|---|
| `Bkpj1zRO` | **0.0000** | dead — reads zero |
| `4VAl2p9k` | **0.0000** | dead — reads zero |
| `95pM7BAV` | **0.0014** | dead — effectively zero |
| `QgbGldAY` | 0.2881 | low |
| `rqAkmpKG` | 0.3581 | low |
| `1Zb102pg` | 0.4448 | low |
| `Rjz2dbXW` | 0.5956 | low |
| `nDby4YpR` | 0.7063 | plausible |
| `3nzr8bGo` | 0.9084 | plausible |
| `wkAWLzlm` | 1.2204 | high |
| `3nzr48bG` | 1.2945 | high |

Three of eleven stations read essentially zero wind at all times. Consequences:

1. **The NWP comparison at those stations is not a forecast comparison.** Scoring a real
   model against a sensor that reads 0.0 registers the model as maximally wrong.
2. The corpus-wide factor spread runs 0.0 → 1.2945. A spread that includes zeros is not a
   calibration spread.
3. Pearson r between station and NWP wind is **0.358**, and the best lag correlation is
   **0.359 at a 1-hour lag**. That is weak enough that "we beat every NWP model on wind"
   should be read as partly a statement about three broken sensors.

Mitigation in place: `build_forecast_windows` accepts `wind_excluded_stations`, which keeps a
station's windows but blanks its three wind feature columns with the **training-set mean**
(not zero, so the network is not taught that calm is the dominant case) and flags the window
so the trainer masks the wind head's loss. **Verify these three IDs are in the exclusion list
before trusting any wind number** — the mitigation is opt-in, not automatic.

### 4.2 Station `wkAWlzlm` returns zero rows

`wkAWlzlm` is on the model station roster (`fetch_current_telemetry.py: MODEL_STATIONS`,
16 stations) and contributes 835 hourly records to the committed corpus, but it has **no
entry in `data/kloudtrack_history_cache/`** — 15 cache files exist for 16 rostered stations,
and `wkAWlzlm` is the missing one. The live
`/api/v1/telemetry/station/wkAWlzlm/history` endpoint answers with zero rows.

This is precisely the failure `check_data_freshness.py` was written for:
`fail_on_missing_station: True`, with the documented rationale *"A rostered station with no
rows at all is the failure this monitor was written for: one of sixteen stations answering
with nothing, in silence."*

Impact: historical training and benchmarking are unaffected (the committed CSV has the
data), but **any fresh refetch that trusts the live endpoint will silently lose this
station.** Run the freshness check before every refetch. Do not assume a successful HTTP 200
means data.

### 4.3 Live surface pressure is unavailable from NOAA GFS

`src/external_sources/gfs.py` serves temperature (TMP 2 m), humidity (RH 2 m) and wind
(UGRD/VGRD 10 m, already in m/s). **It cannot serve pressure.** Every product in the GFS
bucket used here is a `pgrb2` file whose only pressure field is `PRMSL` — the *perturbation*
of mean sea level pressure. Absolute MSL is not published there, and a perturbation is not
a surface pressure: adding a guessed base state injects a constant altitude bias straight
into the per-station correction, which is the exact error those coefficients exist to remove.

So the provider deliberately returns no pressure, the router degrades the pressure cells to
the LNN, and the measured GFS pressure gain (~+30% at 6 h) is **forgone rather than faked**.
This is a correct decision, recorded as a known gap rather than a bug.

### 4.4 Data quality summary (run of record)

From `data/data_quality_report.json`: 756,156 raw weather rows across 16 stations, 12,537
resampled station-hours, range 2026-06-20T10:00 → 2026-08-26T02:00.

| Signal | Status |
|---|---|
| UV index | `BLOCKED_BY_SENSOR_CALIBRATION` — reports up to 11.0 at midnight. Never scored. |
| Light intensity | `SECONDARY_BETA_DAYLIGHT_ONLY` — uncalibrated photometric sensor, no orientation metadata |
| Water level | `INTERNAL_EXPERIMENT_BETA`, `not_for_life_safety: true`, excluded from the commercial core |
| Temperature / humidity | `FEASIBLE`, 93.4% coverage |
| Pressure | `FEASIBLE`, 92.9% coverage |
| Wind speed | `FEASIBLE`, 92.2% coverage, 57% calm hours |

22,955 weather rows quarantined. Raw field maxima confirm the quarantine is doing real work:
pressure max 288,522 hPa, wind max 3,069 km/h, precipitation max 7,202 mm, heat index max
5,624 °C.

---

## 5. Cells with no skill

Every cell where the served forecast provides no measurable value over persistence:

| Cell | Served | vs persistence | Why |
|---|---|---|---|
| humidity, **all** horizons | persistence | identical (≤0.1%) | no skill found; headroom exists but the promotion rule was not met (§2) |
| pressure, **all** horizons | persistence | identical (~1e-4 hPa) | persistence beats a 0.25° grid cell at surface pressure; the head adds nothing |
| wind_direction, **all** horizons | persistence | not MAE-benchmarked | circular variable; served from origin at every horizon |
| temperature +1h | persistence | −0.02% | persistence is strong at 1 h; the head is −0.45% at +1h |
| temperature +3h | persistence | −0.01% | same |
| temperature +24h | persistence | 0.00% (identical) | see §3 |
| wind_speed +1h | persistence | −0.02% | head is −0.45%; genuinely no skill at 1 h |
| wind_speed +3h | persistence | −0.01% | head is +0.56%, below the 2% promotion margin |
| wind_speed +24h | learned_model | −0.14% | learned but negligible; the network is close to persistence at 24 h |

For pressure and humidity, "no skill" is not a failure of the search — it is the correct
finding. A 1-hour-ahead pressure forecast that copies the barometer is very hard to beat,
and a 0.25° model cell is a weak comparator for station-level surface pressure over terrain.
The honest summary is: **the system is not forecasting pressure or humidity, and it does not
claim to.**

---

## 6. Bug history — seven defects, each of which produced a wrong number that was believed

Documented because they are the reason the current numbers can be trusted, and because each
one produced a plausible-looking figure that a reasonable person had no reason to doubt.
Every fix is recorded alongside the check that would have caught it.

### 6.1 NWP wind units (km/h vs m/s)

- **Defect.** Open-Meteo's `wind_speed_10m` is served in km/h; stations report m/s. They were
  compared directly.
- **Wrong figure produced.** Every NWP wind error inflated by **3.6×**. The system appeared
  roughly **6× better than ECMWF** when it was actually about **1.4× better**.
- **Why it was believed.** The direction of the error flatters the system, and a suspiciously
  good result is easier to accept than a mediocre one.
- **Fix.** `NWP_TO_STATION_UNITS = {"wind_speed": 1.0 / 3.6}`, applied to the Open-Meteo
  fields only. The candidate model already predicts m/s and must **not** have the factor
  applied — doing so rescales the candidate by 3.6× the other way.
- **Check.** `benchmark_vs_nwp.py:77–81` records the verification: station mean 1.501 m/s
  versus NWP 11.19 km/h = 3.11 m/s.
- **Residue.** `data/nwp_benchmark_report.txt` is a pre-fix artifact and still reports wind in
  km/h with NWP errors of 7–10. See §8.

### 6.2 NWP window selection (`load_nwp` kept only the longest window)

- **Defect.** The cache is keyed `station|model` *or* `station|model|<start>_<end>`; the same
  series can be cached under several fetched windows. `load_nwp()` kept only the **longest**
  per station. The longest cached series ended 2026-08-28, while a newer targeted fetch
  covered the test split from 2026-09-11 onward — so longest-wins selected a window with
  **zero overlap with the evaluation period**.
- **Wrong figure produced.** Every NWP lookup missed, **all seven models printed `n/a`**, and
  the report still printed a **rank** as if it had scored them. A table with N/A rows and a
  confident ordering beneath it is worse than an error, because it invites the reader to
  ignore the rows that failed.
- **Fix.** Merge on timestamp. Shortest windows are applied **first** and longer ones fill
  only timestamps still missing, so a narrow recent re-fetch wins where it overlaps and no
  coverage is lost elsewhere. Within equal spans, the later-ending window wins.
- **Subtle residue.** The first merge attempt used an *unconditional* write instead of
  fill-if-missing, which silently reproduced longest-wins through a timestamp map where it
  was no longer visible to review. The `continue` on already-filled fields is load-bearing.
- **Check to run every time.** A model row that reads `n/a` must stop the run. There is no
  legitimate reason for a benchmark to print a rank alongside missing scores.

### 6.3 Candidate feature space (scored on the wrong input space)

- **Defect.** Candidate architectures were scored on **de-normalised, clipped** input when
  they had been trained on **normalised** input.
- **Wrong figure produced.** +12h temperature **degraded by 35.7%**, which was read as "the
  feature-augmented candidate is worse at +12h" and used to reject the architecture.
- **Why it was believed.** A 35.7% degradation is large enough to look like a real finding
  about the architecture rather than a harness bug.
- **Fix.** The candidate path must be fed normalised input. Production bundles de-normalise
  internally; candidates are loaded directly and must not be de-normalised before them.
- **Check.** `benchmark_vs_nwp.py:361–363` documents why `X_raw` is passed to the bundle
  predictor and not to the candidate.

### 6.4 Promotion audit tuple-arity bug

- **Defect.** Each entry of the routing table is a 4-tuple `(label, target, metric, block)`.
  The fourth field was missing from every row, so the unpack raised *"not enough values to
  unpack"*.
- **Wrong figure produced.** The **entire promotion audit was silently skipped on every run,
  including runs that appeared to succeed** — while the run left complete artifacts on disk
  and exited `rc=1`. A run that produces valid-looking artifacts *and* a non-zero exit code
  tells you both that something worked and that something did not; the natural reading is
  that the non-zero code is the minor part.
- **Why it was believed.** The artifacts were complete. There was no field anywhere saying
  "promotion audit not run."
- **Fix.** All four fields present. Partial runs (`--horizons` without 1 h) now set
  `promotion_audit: None` **explicitly** with a note, rather than leaving the key absent.
- **Check to run every time.** If `promotion_audit` is missing from the run output, the run
  did not evaluate promotion. `rc != 0` is never ignorable.

### 6.5 Bias sign (two conventions, opposite answers)

- **Defect.** `benchmark_vs_nwp.metrics()` reported `mean(truth − pred)` while `monitoring.py`
  (the operational dashboard) reported `mean(pred − truth)` — the WMO convention.
- **Wrong figure produced.** The **same forecast read as over-predicting on one screen and
  under-predicting on the other.** Anyone triaging a systematic bias from the dashboard would
  have worked on the wrong end of the model.
- **Why it was believed.** Neither number was wrong on its own terms. Both were internally
  consistent. There is no way to catch this from a single screen.
- **Fix.** `src/scoring.py` is now the single canonical scorer: `bias = mean(pred − truth)`,
  positive meaning the model runs warm. Golden fixtures pin the sign by assertion.
- **Known remaining exception.** `benchmark_independent.score` still uses `mean(truth − pred)`,
  and `diagnose_headroom.mae` returns `mean(truth − pred)` as its second element. **Neither is
  a reportable bias figure.** `scoring.py` documents the table; do not infer it.

### 6.6 Wind head dead gradient (ReLU floor)

- **Defect.** `F.relu` on the wind output has exactly zero gradient on its negative side.
- **Wrong figure produced.** The head learned a **−5.75 m/s** residual and **70% of test
  windows clamped to 0.0** with no gradient path back. Wind forecasts read 0.0 for the
  majority of windows — a plausible-looking calm forecast that was actually a dead unit.
- **Why it was believed.** Zero wind is not an obviously invalid output. It passes a
  plausibility check, a bounds check, and renders cleanly on a dashboard.
- **Fix.** `_nonnegative()` — `F.leaky_relu` with slope 0.01, then `clamp(max=…)` for the
  physical ceiling only. The floor lives inside the residual addition; an outer `clamp(min=0)`
  would restore the dead gradient.
- **Check.** `src/test_wind_head_gradient.py` asserts the gradient is non-zero on the
  negative side.

### 6.7 Seed-sweep arms that measured nothing (≈6 h of compute)

- **Defect.** The comparison harness wrote a per-arm `model.py` into a temporary directory
  and then launched the **repository's** trainer, which imports `model.py` as a sibling module
  — so **both arms trained the same model**.
- **Wrong figure produced.** Two "different" arms produced scores identical to **six decimal
  places**, presented as a controlled comparison of two architectures. Roughly six hours of
  compute measured nothing.
- **Why it was believed.** Identical numbers usually mean a deterministic pipeline working
  correctly.
- **How it was caught.** **Identical standard deviations across independently-trained models is
  impossible.** Two runs with different seeds and different architectures cannot produce
  matching variance. That single inconsistency gave it away.
- **Fix.** The trainer must be launched **from the arm directory**, not from the repo root.
  This is now stated in the docstring of `run_arm()` so the mistake cannot be repeated
  silently.
- **Check to run every time.** Across independently-seeded arms, the per-seed standard
  deviations must differ. If they match exactly, the arms are not independent and the sweep
  measured nothing.

---

## 7. Rain occurrence: the strongest result, and its decay

Rain is the one channel where the learned model participates at every horizon, and the only
one where the served forecast ranks 1st of 10. The full ranking, from
`data/nwp_benchmark_production.json`:

| Horizon | Rank | vs best NWP (DWD ICON) | vs persistence | vs climatology |
|---|---|---|---|---|
| +1h | **1 / 10** | +43.2% | +15.9% | +44.0% |
| +3h | **1 / 10** | +23.0% | +12.8% | +24.1% |
| +6h | **1 / 10** | +9.0% | +10.1% | +9.3% |
| +12h | **1 / 10** | +2.1% | +10.7% | +2.5% |
| +24h | **3 / 10** | **−2.7%** | +13.5% | **−4.6%** |

Two failure modes of reporting this channel, both of which have occurred:

1. **Quoting the +1h margin without the horizon.** The margin over the best NWP model decays
   43.2% → 23.0% → 9.0% → 2.1% across the horizon range. The +12h win is real but thin:
   0.2364 against DWD ICON's 0.2415 is inside the run-to-run noise of any calibration change.
   "+43% better than ECMWF on rain" is true at +1h and false at +6h, +12h and +24h.
2. **Saying "beats all seven NWP models at every horizon."** DWD ICON beats the system at
   +24h. So does climatology. The rank at +24h is 3rd of 10, not 1st.

The honest summary: **a genuine short-lead win that decays fast and does not survive
+24h.** It is also the only channel whose strongest competitor margin is shrinking while
the system's own persistence margin stays roughly flat (+15.9% → +13.5%) — which suggests
the learned occurrence signal degrades with lead faster than the persistence signal does,
exactly as the blend weights in `inference_policy.json` already assume.

---

## 8. Stale artifacts that will mislead you

Several documents in this repository carry pre-fix numbers. They are not marked stale in their
own text. Check what you are quoting.

| Artifact | What is wrong |
|---|---|
| `prediction-model/README.md` §4 | Older scorecard: rain Brier 0.1219 at +1 h (current: 0.1373), pressure "+6h 1.10 vs 1.11", wind quoted in **km/h**. Superseded by `nwp_benchmark_production.json`. |
| `data/nwp_benchmark_report.txt` | **Pre wind-units-fix.** Wind in km/h with NWP errors of 7–10. Contains the inflated figures from §6.1. Not a source of record. |
| `data/nwp_benchmark_results.json` | Different window set from the production benchmark (h24 persistence 1.0784 vs 1.4059). Not interchangeable with `nwp_benchmark_production.json`. |
| `data/nwp_benchmark_candidate_*.json` | Candidate runs, not production. |
| `data/nwp_bm_ctrl_*.json` | ReLU / leaky-floor controls. Useful for the floor comparison only. |
| `MODEL_REGISTRY.md` scorecards | Written against earlier runs. Verify any number against `nwp_benchmark_production.json` before repeating it. |

**The single source of record for accuracy is `data/nwp_benchmark_production.json`.**

---

## 9. What is not claimed

Stated explicitly so that nobody has to infer it:

1. **This system is not competitive with ECMWF or GFS as a general weather forecast.** It
   loses to ECMWF on temperature at +6h and +12h, and ECMWF has the lowest MAE of any method
   at +1h and +3h.
2. **It does not forecast humidity or pressure.** Those cells serve persistence.
3. **Its wind advantage is measured against a ground truth with three dead anemometers.**
4. **Its rain advantage decays to nothing by +12h and reverses at +24h.** It ranks 1st of 10
   from +1h to +12h, but the margin over the best NWP model falls from 43.2% to 2.1%. At +24h
   it ranks 3rd: climatology (0.2396) and DWD ICON (0.2440) both beat it (0.2506).
5. **Its +24h numbers are provisional** until the diurnal-phase question in §3.2 is tested at
   offset horizons.
6. **All results are for a single test window** — roughly 10 days, 2026-08-15 → 2026-08-26,
   across 16 stations in one region. Nothing here establishes seasonal or cross-region
   generalisation.
7. **Not for life safety.** Water level is an internal beta experiment
   (`not_for_life_safety: true`); flood and evacuation claims are excluded from the product
   core.

---

## 10. What would actually change these numbers

Ordered by measured headroom, not by effort.

1. **Open the humidity gate selectively.** The head shows +19.74% at +12h and it is being
   discarded. Either promote per-horizon rather than per-variable, or document why a 19.74%
   single-horizon improvement should not count.
2. **Test +24h at offset horizons** (+18h, +30h) to settle the diurnal-phase question in §3.2
   before any +24h claim is made externally.
3. **Exclude the three dead anemometers from the benchmark** and re-run. The wind margin
   against NWP will change, probably by a lot, and it needs to be the real margin.
4. **Investigate `wkAWlzlm`** — repair the endpoint or remove it from the roster. Do not leave
   a station that answers with nothing on a roster that a freshness check will fail on.
5. **Pressure from GFS** is forgone, not solved. Absolute MSL is not available from the
   product bucket used. Revisit only with a product that publishes it.
6. **Re-run the seed sweep correctly** (with per-seed standard deviations verified to differ)
   to size the run-to-run noise band. Without it, no effect smaller than that band can be
   promoted, and several headroom numbers above may be inside it.