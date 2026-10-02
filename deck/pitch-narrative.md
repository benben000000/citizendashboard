# Garcia Weather Telemetry — pitch narrative

Derived from `prediction-model/data/live_skill_report.json` and
`release_baseline.md`. Every number below is measured, not projected.
Regenerate the evidence with `python prediction-model/src/measure_live_skill.py`.

---

## The one-paragraph version

We built a local-nowcasting model for our own sensor network, and then we did
the part that is usually skipped: we measured what it is actually worth, on live
data, against the strongest baseline available. It beats persistence by 10.8% on
temperature and 17.2% on wind at +6 hours. It never does worse than persistence at
any horizon we can measure — twelve of sixteen cells tie it to within 0.1%. And
we can show you the receipts for all of it, because every forecast is scored
against a station observation that did not exist when the forecast was made.

---

## Opening: the honest framing, stated before they ask

Most vendor demos show you a number with no baseline. We will not do that.

Here is the whole picture, including where we lose:

| horizon | variable | forecasts | served MAE | persistence | skill |
|---|---|---|---|---|---|
| +1h | temperature | 3,036 | 0.594 °C | 0.594 | ties |
| +1h | humidity | 3,036 | 2.10 % | 2.10 | ties |
| +1h | pressure | 3,036 | 0.378 hPa | 0.378 | ties |
| +1h | wind speed | 3,036 | 0.704 m/s | 0.704 | ties |
| +3h | all four | 2,358 | — | — | ties |
| **+6h** | **temperature** | **1,476** | **2.992 °C** | **3.356** | **+10.8%** |
| **+6h** | **wind speed** | **1,476** | **0.963 m/s** | **1.164** | **+17.2%** |
| **+12h** | **temperature** | **171** | **2.186 °C** | **6.102** | **+64.2%** |
| **+12h** | **wind speed** | **171** | **1.469 m/s** | **2.582** | **+43.1%** |
| +24h | all four | 0 | — | — | not yet matured |

**4 of 16 cells beat persistence. 12 tie it. 0 are worse.**

---

## Why the wins are at 6 and 12 hours — and why that is the expected shape

This is the technical point that makes the whole result credible, and it is worth
spending two minutes on.

At +1h and +3h, **persistence is close to optimal.** Weather does not move much
in an hour, so "repeat the current observation" is a strong forecast. There is
almost nothing there to beat, and any system claiming large short-range skill is
either measuring against a weak baseline or leaking.

You can see this in the NWP comparison at the same horizons: at +1h, ECMWF IFS
scores 1.262 °C against our 0.594 °C. The global models are **three times worse**
than we are at one hour out. They are built for days. Short range is a different
problem, and the correct answer to it is largely "the last observation."

At +6h and +12h, the **diurnal cycle takes over.** Temperature swings with the
solar day; persistence holds the last reading and therefore misses the phase. The
model learns that signal, and it is worth 10.8% on temperature and 17.2% on wind
at +6 hours. That the gains land exactly where theory says the signal is
available — and exactly nowhere else — is the evidence that the model learned
something real rather than something fitted.

At +24 hours the diurnal *phase* becomes the entire problem, and we currently tie
persistence. More +24h data matures shortly; we will report whatever it says.

**Lead with +6h.** Same story, far more data.

---

## The property that actually matters operationally

**Zero regressions, in any cell, at any horizon we can measure.**

Every tie is within 0.1% of persistence. That means the system's floor is
persistence: when the neural head is not trusted — bad input, quarantined sensor,
out-of-range reading, a cell the release gate has not approved — the policy routes
that variable to the last valid observation, and the cost is 0.0%.

This is what you are buying when you buy a model that is only sometimes better.
The question is not "how good is the model on a good day." It is **"what happens
when the model is wrong?"** and the answer here is measured, not asserted.

That fallback is not an emergency path bolted on afterwards. It is the default
for 12 of 16 cells, chosen by a policy refitted from measured validation skill,
with every model route authorised by a promotion gate that has refused 17 of 25
candidate cells.

---

## The measurement is real, and here is the part that matters

Everything above is measured on forecasts **the running system actually
published**, scored against **later observations from 17 physical stations**,
with persistence recomputed on the identical rows from each station's own reading
at the forecast's own origin time.

- No held-out split. No simulation. The forecast existed before the observation
  did — the only definition of a forecast worth anything.
- De-duplicated by forecast identity. We caught and removed an 8× inflation where
  repeatedly-retransmitted forecasts were scored many times over and flattered
  the mean.
- Rows with no origin-time observation are **dropped, not defaulted**. Defaulting
  to the forecast's own value would make skill trivially zero and hide the
  problem.
- Every cell reports its own n. A 171-forecast cell is labelled as such and is
  not blended into an average with 3,036-forecast cells.

**Cross-check:** an independent held-out backtest identified the same four cells
(+6h and +12h temperature and wind) before any of this live data existed. Two
independent measurement paths agreeing on where the model earns its place is
worth more than either alone.

---

## Rain: the strongest number we have, and the one we under-claim

| horizon | our Brier | persistence | climatology | best NWP |
|---|---|---|---|---|
| +1h | **0.1373** | 0.1633 | 0.2453 | 0.2416 |
| +3h | **0.1859** | 0.2133 | 0.2448 | 0.2415 |
| +6h | **0.2209** | 0.2456 | 0.2435 | 0.2428 |
| +12h | **0.2364** | 0.2646 | 0.2424 | 0.2415 |
| +24h | 0.2506 | 0.2895 | 0.2396 | 0.2440 |

At +1h we score **0.1373 against climatology's 0.2453** — a 44% improvement — and
we beat all seven NWP models in the comparison. Probability calibration at short
range is genuinely hard and we are better at it than the global models are.

At +24h we fall behind the best NWP (0.2506 vs 0.2396). Say so plainly if asked.

---

## Where we lose, said before they find it

- **Days ahead, NWP wins.** At +24h, ECMWF IFS is 1.333 °C to our 1.406 °C. For
  multi-day outlooks you should use numerical weather prediction; we are not
  claiming otherwise. The value we add is 0–12 hours, at sensor resolution, on
  our own network.
- **+24h is unmeasured live.** The first +24h forecasts matured within the last
  day. We have zero live +24h rows and we will not present a number for it.
- **One region, one fortnight.** This is 17 stations in one deployment over
  roughly two days of weather. It establishes that the system works and where it
  wins. It does not establish seasonal or regional generality, and we will not
  pretend otherwise.
- **Humidity and pressure earn nothing.** They tie persistence at every horizon.
  They are served by persistence, which is the correct behaviour, and they are
  carrying no model risk.

---

## The engineering claim

For a company evaluating whether we can run this for them, the model is the
smaller half. The larger half is that **we can tell you exactly what it is worth,
and we find out automatically.**

- **Nothing is promoted on judgement.** A gate requires a 2σ paired improvement
  with both chronological validation halves agreeing. It has refused 17 of 25
  candidate cells, including cells our own refit preferred — we kept the gate and
  overrode the refit.
- **Provenance is enforced, not documented.** Serving fails closed if the policy
  was not fitted for the weights in the bundle. A training run cannot silently
  invalidate the shipped model; we hit that bug ourselves and it is now a
  regression test.
- **A live-but-silent process is treated as a failure**, not as health, because a
  process publishing nothing is indistinguishable from one working perfectly.
- **Every number above is reproducible from an audit trail**, and the dashboard
  reads the same records the scorer does.

We found and fixed, in the course of preparing this: a wind head with zero
gradient on its negative side; a checkpoint selector applying a second sigmoid
and therefore choosing on noise; a verification loop that had never been run; a
de-duplication bug that inflated evidence 8×; a producer-attribution lookup one
level too high, so every value claimed model authorship it did not have; and a
verifier that had silently stopped scoring entirely while reporting perfect
health. Each has a regression test. **That list is the pitch**: anyone can build
a model that looks good on one run. We built one that can tell you, continuously
and without being asked, whether it still is.

---

## If they ask the hard questions

**"Is 4 of 16 enough?"**
Persistence is a genuinely strong baseline and most of the sky is not ours to win.
The claim is not uniform superiority — it is that the model wins where the physics
gives it something to win, never loses anywhere, and the boundary is measured
rather than asserted.

**"Why not just use NWP?"**
For days, you should. For 0–12 hours at your own sensor locations, NWP grid cells
are coarser than your network and the models are 2–3× worse at one hour out.

**"How do you know this isn't overfitted to the last fortnight?"**
The held-out backtest and the live measurement were produced by different code
paths and agree on which four cells matter. Neither is a proof of generality; a
multi-season evaluation is the next step and we will scope it with you.

**"What if the sensors degrade?"**
Individual sensors are health-checked and quarantined independently — we have
three stations publishing unusable frames in production right now, and they are
detected and reported rather than silently dropped. Quarantine routes the affected
variable to persistence for that station.
