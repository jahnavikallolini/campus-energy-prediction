# Smart Campus Energy Consumption Prediction and Anomaly Detection (I-BLEND)

B.Tech mini-project (CSE AI&ML, Manipal Institute of Technology). Hourly prediction of the electrical power of seven campus
building feeders (Linear Regression, KNN, Decision Tree, SVR), residual-based detection of *potential anomalous consumption
periods*, PCA/K-Means exploration, a Streamlit dashboard, an IEEE-format report and a presentation.

## Read this first: known data issue
Energy timestamps appear displaced by about 4.5-5.5 h from roughly 2017-07-08 in all seven series (occupancy is not affected).
The cause is unresolved and **nothing was corrected**. The primary test period therefore ends on 2017-07-07; later rows are a
sensitivity analysis. Details: `reports/timestamp_investigation.md`. Flagged anomalies are statistical outliers relative to the
model, **not confirmed physical faults**.

## Deliverables
| What | Where |
|---|---|
| IEEE-format report (LaTeX source + PDF) | `reports/ieee_report/main.tex`, `main.pdf` |
| Presentation (10 slides) | `presentation/campus_energy_presentation.pptx` |
| Dashboard | `app/streamlit_app.py` |
| Dataset audit, preprocessing report, model evaluation report, timestamp investigation | `reports/*.md` |
| Results | `results/tables`, `results/metrics`, `results/figures`, `results/predictions`, `results/anomalies` |
| Saved models (per variant, model, building) | `models/*.joblib` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Python 3.12 was used
pip install -r requirements.txt
```
Place the raw I-BLEND files as described in `data/README.md` (or `export IBLEND_RAW_DIR=/path/to/raw`). Raw files are never modified.

## Run the whole pipeline
```bash
./run_all.sh      # audit -> hourly preprocessing -> features/split -> leakage tests -> Lecture diagnostics -> EDA -> preprocessing report
./run_models.sh   # validation-only tuning -> frozen-model evaluation -> model-stage leakage tests -> figures -> timestamp diagnostic -> model report
./run_final.sh    # test periods -> anomaly detection + plots -> PCA/K-Means -> dashboard smoke test -> report and deck builders
```
`run_models.sh` takes about 15 minutes on one CPU core (SVR tuning dominates); checkpoints in `results/tuning_checkpoints` and
`results/eval_checkpoints` make an interrupted run resumable (delete them to recompute from scratch).

## Dashboard
```bash
streamlit run app/streamlit_app.py
python -m tests.smoke_app        # headless smoke test (AppTest)
```
Sidebar: building, model (SVR primary or Linear Regression), feature set (energy + lagged occupancy, or energy only), date range and a check box for the post-break
rows. Tabs: overview, actual vs expected, metrics, anomalies (timeline, events, thresholds), CSV export. It reads saved outputs only; the "saved model" button re-runs the
model files in `models/` on the selected hours (stored predictions are rounded to 6 significant digits, so differences of up to about 1 W are expected).

## Method summary (all settings in `src/config.py`)
* Target: hourly mean of valid 1-min `power` (W; valid if >= 45 of 60 readings); seven main feeders only (hostel UPS feeders excluded).
* Features: hour, day of week, weekend, month, working_day, activity_high, power lags 1/2/24, 24 h rolling mean/std (t-24..t-1), occupancy lags 1/2/24.
  Missing occupancy stays missing (never zero); targets are never filled; lag inputs may use causal carry-forward of at most 1 h. Weather excluded (2018 only).
* Split: train 2014-02-17 to 2016-06-30; validation 2016-07-01 to 2016-12-31; primary test 2017-01-01 to 2017-07-07; sensitivity: 2017-07-11 onward.
* Models: one per building; encodings and hyper-parameters chosen on validation only; frozen models fitted on training rows only; predictions clipped at 0 W.
* Anomalies: |modified z| > 3.5 on residuals; median/MAD from validation residuals per model, feature set, building, day-part and regime (`src/anomaly/detect.py`).
* PCA on training rows (exploratory); K-Means on building-day profiles (selected by silhouette).

## Tests
`python -m tests.test_features` (feature leakage), `python -m tests.test_models` (lags, rolling windows, scalers, encodings, tuning isolation, split), `python -m tests.smoke_app` (dashboard).

## Structure
```
app/            Streamlit dashboard            src/config.py     all settings
src/data/       audit, loaders, preprocessing  src/features/     features + split
src/models/     tuning, evaluation             src/anomaly/      detection + plots
src/analysis/   EDA, diagnostics, PCA/K-Means, report and deck builders
tests/          leakage and smoke tests        presentation/     deck builder, data and .pptx
reports/        audit, evaluation, timestamp investigation, IEEE report
```
Rebuilding the report needs `pdflatex`, `bibtex` and the IEEEtran class; the deck builder needs Node.js with `pptxgenjs` (see `presentation/build_deck.js`).

## Limitations
No ground truth for anomalies; timestamp discontinuity (above); occupancy is a Wi-Fi proxy; Lecture is zero-inflated (about 80% of hours at 0 W) and its metrics and flags are less reliable;
SVR's selected hyper-parameters are at the edge of the searched grid and its margin over Linear Regression is small; one campus only. PCA/K-Means are exploratory (K-Means on seven buildings is dominated by Lecture).
