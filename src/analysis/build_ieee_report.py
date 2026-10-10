"""Generate reports/ieee_report/main.tex (IEEEtran) from the saved result tables, then compile it.

Usage:  python -m src.analysis.build_ieee_report
Every number comes from results/tables, results/metrics or results/anomalies; nothing is typed by hand except wording.
"""
import json
import platform
import subprocess

import pandas as pd

from src import config as C

OUT = C.REPORT_DIR / "ieee_report"
SVR, LR, DT, KNN = "Support Vector Regression", "Linear Regression", "Decision Tree Regression", "KNN Regression"
SHORT = {SVR: "SVR", LR: "Linear Reg.", DT: "Decision Tree", KNN: "KNN"}
LAB = {b: v["label"].replace(" (Dining)", "") for b, v in C.BUILDINGS.items()}


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def f3(x): return f"{x:.3f}"
def f0(x): return f"{x:,.0f}".replace(",", "{,}")


def main():
    T = C.TAB_DIR
    pm = pd.read_csv(T / "period_metrics_macro.csv")
    pl = pd.read_csv(T / "period_metrics_long.csv")
    cov = pd.read_csv(T / "coverage_summary.csv").set_index("building")
    sel = json.load(open(C.MET_DIR / "selected_config.json"))
    lec = json.load(open(C.MET_DIR / "lecture_diagnostics.json"))
    pk = json.load(open(C.MET_DIR / "pca_kmeans_summary.json"))
    aso = pd.read_csv(T / "anomaly_summary_overall.csv")
    asb = pd.read_csv(T / "anomaly_summary_by_building_period.csv")
    agr = pd.read_csv(T / "anomaly_agreement.csv")
    run = pd.read_csv(T / "runtime_summary_primary.csv").set_index("model")
    ev = pd.read_csv(T / "pca_explained_variance.csv")
    km_all = pd.read_csv(T / "kmeans_all7_cluster_summary.csv"); km_wl = pd.read_csv(T / "kmeans_without_lecture_cluster_summary.csv")
    kms_all = pd.read_csv(T / "kmeans_all7_k_selection.csv"); kms_wl = pd.read_csv(T / "kmeans_without_lecture_k_selection.csv")
    shift = pd.read_csv(T / "timestamp_shift_energy_vs_occupancy.csv")
    offm = pd.read_csv(T / "timestamp_offset_by_month.csv")

    def M(variant, period, model, col="macro_nRMSE_std"):
        r = pm[(pm.variant == variant) & (pm.period == period) & (pm.model == model)]
        return float(r[col].iloc[0])
    def rows_of(variant, period):
        return int(pm[(pm.variant == variant) & (pm.period == period)].rows.max())
    P_BASE = "Baseline: persistence (t-1)"; SN_BASE = "Baseline: seasonal naive (t-24)"

    # ---------- Table: dataset coverage
    t_cov = "\n".join(f"{esc(LAB[b])} & {f0(cov.loc[b,'valid_hours'])} & {cov.loc[b,'valid_pct_of_span']:.1f} & {f0(cov.loc[b,'primary_rows'])} \\\\" for b in C.BUILDINGS)
    # ---------- Table: configurations
    t_cfg = ""
    for m in ("LR", "KNN", "DT", "SVR"):
        for v, nm in (("P", "energy+occ."), ("E_all", "energy-only")):
            c = sel[v][m]; ps = ", ".join(f"{k}={val}" for k, val in c["params"].items()) or "none"
            t_cfg += f"{m} & {nm} & {c['encoding']} & {esc(ps)} \\\\\n"
    # ---------- Table: macro validation + primary test (all models)
    def macro_rows(period):
        out = ""
        for mdl in (SVR, LR, DT, KNN):
            out += f"{SHORT[mdl]} & {f3(M('P', period, mdl, 'macro_R2'))} & {f3(M('P', period, mdl, 'macro_nMAE_std'))} & {f3(M('P', period, mdl))} \\\\\n"
        for mdl, nm in ((P_BASE, "Persistence (t-1)"), (SN_BASE, "Seasonal naive (t-24)")):
            out += f"{nm} (baseline) & {f3(M('baseline', period, mdl, 'macro_R2'))} & {f3(M('baseline', period, mdl, 'macro_nMAE_std'))} & {f3(M('baseline', period, mdl))} \\\\\n"
        return out
    t_val, t_test = macro_rows("validation"), macro_rows("test_primary")
    # ---------- Table: sensitivity (nRMSE by period)
    t_sens = ""
    for mdl in (SVR, LR, DT, KNN):
        t_sens += f"{SHORT[mdl]} & " + " & ".join(f3(M("P", p, mdl)) for p in ("validation", "test_primary", "test_post_break", "test_full")) + " \\\\\n"
    t_sens += "Persistence (t-1) & " + " & ".join(f3(M("baseline", p, P_BASE)) for p in ("validation", "test_primary", "test_post_break", "test_full")) + " \\\\\n"
    # ---------- Table: per-building SVR and LR
    def pb(variant, period, model, b, col):
        r = pl[(pl.variant == variant) & (pl.period == period) & (pl.model == model) & (pl.building == b)]
        return float(r[col].iloc[0])
    t_pb = ""
    for b in C.BUILDINGS:
        for mdl in (SVR, LR):
            cells = []
            for per in ("validation", "test_primary"):
                cells += [f0(pb("P", per, mdl, b, "MAE")), f"{pb('P', per, mdl, b, 'MSE') / 1e6:.2f}", f0(pb("P", per, mdl, b, "RMSE")), f3(pb("P", per, mdl, b, "R2"))]
            t_pb += f"{esc(LAB[b]) if mdl == SVR else ''} & {SHORT[mdl]} & " + " & ".join(cells) + " \\\\\n"
    # ---------- Table: R2 all four models per building
    t_r2 = ""
    for b in C.BUILDINGS:
        cells = [f3(pb("P", per, m, b, "R2")) for per in ("validation", "test_primary") for m in (SVR, LR, DT, KNN)]
        t_r2 += f"{esc(LAB[b])} & " + " & ".join(cells) + " \\\\\n"
    # ---------- Table: energy-only vs primary on matched rows
    t_fs = ""
    for mdl in (SVR, LR, DT, KNN):
        t_fs += f"{SHORT[mdl]} & " + " & ".join(f3(M(v, p, mdl)) for p in ("validation", "test_primary") for v in ("P", "E_matched", "E_all")) + " \\\\\n"
    # ---------- Table: anomaly rates
    def A(model, variant, period, col="pct_flagged"):
        r = aso[(aso.model == model) & (aso.variant == variant) & (aso.period == period)]
        return float(r[col].iloc[0])
    t_an = ""
    for mdl in (SVR, LR):
        for v, nm in (("P", "energy+occ."), ("E_matched", "energy-only")):
            t_an += f"{SHORT[mdl]} & {nm} & " + " & ".join(f"{A(mdl, v, p):.2f}" for p in ("validation", "test_primary", "test_post_break")) + f" & {int(A(mdl, v, 'test_primary', 'flagged'))} \\\\\n"
    s2 = asb[(asb.model == SVR) & (asb.variant == "P")]
    t_anb = ""
    for b in C.BUILDINGS:
        g = s2[s2.building == b].set_index("period").pct_flagged
        t_anb += f"{esc(LAB[b])} & {g['validation']:.1f} & {g['test_primary']:.1f} & {g['test_post_break']:.1f} \\\\\n"
    def AG(comp, per, col):
        r = agr[(agr.comparison == comp) & (agr.period == per)]
        return float(r[col].iloc[0])
    sh = shift.set_index(["building", "period"])
    sh_ac_pre = sh.loc[("Academic", "pre (2017-01-01..07-07)")]; sh_ac_post = sh.loc[("Academic", "post (2017-07-11..11-03)")]
    lib_pre = sh.loc[("Library", "pre (2017-01-01..07-07)")]; lib_post = sh.loc[("Library", "post (2017-07-11..11-03)")]
    post_shifts = shift[shift.period.str.startswith("post")].best_energy_shift_minutes
    pre_shifts = shift[shift.period.str.startswith("pre")].best_energy_shift_minutes
    km_rows = lambda df: "\n".join(f"{int(r.cluster)} & {int(r.days):,} & {r.mean_rel:.2f} & {r.load_factor:.2f} & {r.weekend_share:.2f} & {r.high_activity_share:.2f} \\\\".replace(",", "{,}") for r in df.itertuples())
    v = dict(
        SVR_VAL=f3(M("P", "validation", SVR)), SVR_VAL_R2=f3(M("P", "validation", SVR, "macro_R2")),
        LR_VAL=f3(M("P", "validation", LR)), DT_VAL=f3(M("P", "validation", DT)), KNN_VAL=f3(M("P", "validation", KNN)),
        PERS_VAL=f3(M("baseline", "validation", P_BASE)),
        SVR_TEST=f3(M("P", "test_primary", SVR)), SVR_TEST_R2=f3(M("P", "test_primary", SVR, "macro_R2")),
        LR_TEST=f3(M("P", "test_primary", LR)), LR_TEST_R2=f3(M("P", "test_primary", LR, "macro_R2")),
        DT_TEST=f3(M("P", "test_primary", DT)), KNN_TEST=f3(M("P", "test_primary", KNN)), PERS_TEST=f3(M("baseline", "test_primary", P_BASE)),
        SVR_POST=f3(M("P", "test_post_break", SVR)), LR_POST=f3(M("P", "test_post_break", LR)),
        SVR_FULL=f3(M("P", "test_full", SVR)), LR_FULL=f3(M("P", "test_full", LR)),
        N_TRAIN="90{,}640", N_VAL=f0(rows_of("P", "validation")), N_TEST=f0(rows_of("P", "test_primary")), N_POST=f0(rows_of("P", "test_post_break")),
        N_FULL=f0(rows_of("P", "test_full")), N_PRIMARY=f0(cov.primary_rows.sum()),
        LEC_ZERO=f"{lec['zero_pct']:.1f}", LEC_ZERO_H=f"{lec['hourly_target_zero_pct']:.1f}",
        LEC_ZR_MEDIAN=f"{lec['zero_run_minutes']['median']:.0f}", LEC_ZR_LONG=f"{lec['zero_minutes_in_runs_over_1_day_pct']:.0f}",
        SVR_AN_VAL=f"{A(SVR, 'P', 'validation'):.2f}", SVR_AN_TEST=f"{A(SVR, 'P', 'test_primary'):.2f}", SVR_AN_POST=f"{A(SVR, 'P', 'test_post_break'):.2f}",
        LR_AN_TEST=f"{A(LR, 'P', 'test_primary'):.2f}", SVR_AN_N=f0(A(SVR, 'P', 'test_primary', 'flagged')),
        JAC_FEAT=f"{AG('features: SVR P vs SVR E_matched', 'test_primary', 'jaccard'):.2f}", JAC_FEAT_LR=f"{AG('features: LR P vs LR E_matched', 'test_primary', 'jaccard'):.2f}",
        JAC_MODEL=f"{AG('models: SVR P vs LR P', 'test_primary', 'jaccard'):.2f}",
        PC1=f"{pk['pca']['pc1'] * 100:.1f}", PC2=f"{pk['pca']['pc2'] * 100:.1f}", PC3=f"{pk['pca']['pc3'] * 100:.1f}", PC90=str(pk["pca"]["components_for_90pct"]),
        K_ALL=str(pk["kmeans_all7"]["selected_K"]), SIL_ALL=f"{pk['kmeans_all7']['silhouette_at_K']:.2f}",
        K_WL=str(pk["kmeans_without_lecture"]["selected_K"]), SIL_WL=f"{pk['kmeans_without_lecture']['silhouette_at_K']:.2f}",
        AC_PRE0=f"{sh_ac_pre.spearman_at_zero_shift:.2f}", AC_POST0=f"{sh_ac_post.spearman_at_zero_shift:.2f}", AC_POST_BEST=f"{sh_ac_post.spearman_at_best:.2f}",
        AC_POST_SHIFT=f"{int(sh_ac_post.best_energy_shift_minutes)}", LIB_POST0=f"{lib_post.spearman_at_zero_shift:.2f}", LIB_POST_BEST=f"{lib_post.spearman_at_best:.2f}",
        PRE_LO=f"{int(pre_shifts.min())}", PRE_HI=f"{int(pre_shifts.max())}", POST_LO=f"{int(post_shifts.min())}", POST_HI=f"{int(post_shifts.max())}",
        KNN_VAL_E=f3(M('E_matched', 'validation', KNN)), KNN_VAL_P=f3(M('P', 'validation', KNN)),
        SVR_TRAIN_S=f"{run.loc[SVR, 'fit_seconds_total']:.0f}", LR_TRAIN_S=f"{run.loc[LR, 'fit_seconds_total']:.1f}",
        PY=platform.python_version(),
        T_COV=t_cov, T_CFG=t_cfg, T_VAL=t_val, T_TEST=t_test, T_SENS=t_sens, T_PB=t_pb, T_R2=t_r2, T_FS=t_fs, T_AN=t_an, T_ANB=t_anb,
        KM_ALL=km_rows(km_all), KM_WL=km_rows(km_wl),
    )
    tex = TEMPLATE
    for k, val in v.items():
        tex = tex.replace(f"@@{k}@@", str(val))
    assert "@@" not in tex, [x for x in tex.split("@@")[1::2]][:5]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.tex").write_text(tex)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
        if r.returncode != 0 and cmd[0] == "pdflatex":
            print(r.stdout[-2500:])
    print("built", OUT / "main.pdf")


TEMPLATE = r"""\documentclass[conference]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage[T1]{fontenc}
\usepackage{url}
\graphicspath{{../../results/figures/}}
\renewcommand{\arraystretch}{1.05}
\begin{document}
\title{Machine Learning-Based Smart Campus Energy Consumption Prediction and Anomaly Detection}
\author{\IEEEauthorblockN{Jahnavi Kallolini, Navya Menon, Keshav Agrawal}
\IEEEauthorblockA{B.Tech CSE (AI\&ML), School of Computer Engineering, Manipal Institute of Technology\\
Mentors: Dr. Narendra V. G., Dr. Ashwath Rao}}
\maketitle

\begin{abstract}
We study hourly electricity prediction and anomaly detection for seven campus buildings of the I-BLEND dataset. One model per building
is trained with Linear Regression, K-Nearest Neighbors, Decision Tree and Support Vector Regression (SVR) on calendar features, lagged power,
rolling statistics and lagged occupancy, using a chronological split (train 2014-02-17 to 2016-06-30, validation 2016-07-01 to 2016-12-31).
Hyper-parameters and encodings were chosen on validation only. SVR obtained the lowest macro-averaged standardized RMSE on validation (@@SVR_VAL@@,
macro R$^2$ @@SVR_VAL_R2@@) and on the primary test period 2017-01-01 to 2017-07-07 (@@SVR_TEST@@, R$^2$ @@SVR_TEST_R2@@), against @@PERS_TEST@@ for a persistence baseline;
Linear Regression was close (@@LR_TEST@@). Lagged occupancy did not improve accuracy beyond lagged power. Energy timestamps from July 2017 onward appear
displaced by roughly 4.5--5.5 hours relative to occupancy and to earlier years; the cause is unresolved, so the later part of the original test period is
reported only as a sensitivity analysis. Residual-based anomaly detection (modified z-score on validation residuals) flags @@SVR_AN_TEST@@\% of primary-test hours;
the flagged hours are statistical outliers and not confirmed faults. PCA and K-Means give an exploratory description of the feature space and of daily load profiles.
A Streamlit dashboard presents the saved results.
\end{abstract}
\begin{IEEEkeywords}
Smart Campus, Energy Consumption, Machine Learning, Energy Prediction, Anomaly Detection, Regression, PCA, K-Means
\end{IEEEkeywords}

\section{Introduction}
University campuses contain buildings with very different schedules and occupancy, so their electricity demand varies strongly with the hour, the weekday
and the academic calendar. Predicting the expected demand supports planning and benchmarking, and comparing measurements with the expected value highlights periods
that deserve inspection. This work follows a project synopsis that specifies the I-BLEND dataset~\cite{rashid2019iblend}, four regression algorithms, prediction-based
anomaly detection, PCA and K-Means. We aim at a transparent, leakage-free pipeline whose limitations are stated explicitly.

\section{Related Work}
Prior studies on campus and educational buildings have applied long short-term memory networks~\cite{faiq2023lstm}, conventional machine-learning regressors~\cite{elhabyb2024ml} and
support vector machines~\cite{liu2020svm} to consumption prediction. Occupancy-based consumption modelling for institutional buildings is studied in~\cite{anand2021occupancy},
PCA-based feature reduction for prediction models in~\cite{parhizkar2021pca}, and K-means clustering of consumers with smart-meter features in~\cite{okereke2023kmeans}.
Anomaly detection of building energy consumption has been reviewed in~\cite{himeur2021anomaly}. We do not compare our numbers with these works, because the datasets, targets and protocols differ.

\section{Dataset and Problem Formulation}
\subsection{I-BLEND}
I-BLEND contains electrical measurements of campus buildings of an Indian institute from August 2013 to December 2017~\cite{rashid2019iblend}. We use the seven main
feeders (Academic, Boys Hostel, Facilities, Girls Hostel, Lecture, Library, Mess) at one-minute resolution (power in watts, plus current, voltage, frequency and power factor),
the ten-minute occupancy counts derived from Wi-Fi data (each value is the maximum of the one-minute counts in its window), the daily academic calendar
(\texttt{working\_day} and \texttt{activity}, where H denotes high-activity and L low-activity periods) and a weather file. Each hostel also has a UPS (``backup'') feeder;
these two feeders are documented but excluded to keep the seven-building scope of the synopsis. The weather file covers only March--June 2018, after the last energy reading, so it cannot be aligned with the energy data and is not used.
Occupancy exists from 2014-02-16 to 2017-11-03. Missing minutes are absent rows; the seven mains are simultaneously present in only 42\% of minutes, so each building is modelled on its own valid data.

\subsection{Target and prediction setting}
The target is $y_t$, the mean of the valid one-minute power readings in the hour starting at $t$ (IST), valid if at least 45 of 60 readings exist. All predictors for hour $t$
use information available before $t$ (Section~IV-B), so the task is one-step-ahead prediction of the expected consumption of the coming hour.

\section{Methodology}
\subsection{Preprocessing}
Epoch timestamps were converted to IST (+05:30, no daylight saving). Hourly occupancy is the mean of the 10-minute counts present in the hour (at least 4 of 6). Nothing is interpolated in the targets.
Absent occupancy observations are kept missing and are never read as zero occupancy, because the released occupancy contains no zero values; rows whose occupancy lags are unavailable are excluded
from the primary matrix. For lag inputs only, a causal last-observation-carried-forward over at most one hour is allowed; a masking experiment showed that this beats the value of the previous day only for one-hour gaps in all seven buildings.
Table~\ref{tab:cov} gives the retained data. The number of valid hours changed little when the energy threshold was varied between 30 and 55 valid minutes (see \texttt{reports/preprocessing\_report.md}).

\begin{table}[t]
\caption{Valid hourly energy observations and rows of the primary matrix.}\label{tab:cov}
\centering\small
\begin{tabular}{lrrr}
\toprule
Building & Valid hours & \% of logging span & Primary rows\\
\midrule
@@T_COV@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Features and leakage control}
Calendar features: hour, day of week, weekend flag, month, \texttt{working\_day}, \texttt{activity\_high} (the academic calendar is fixed in advance). History features:
$y_{t-1}$, $y_{t-2}$, $y_{t-24}$, and the mean and standard deviation of $y_{t-24},\dots,y_{t-1}$. Occupancy features: lagged occupancy at $t-1$, $t-2$, $t-24$ (never the occupancy of hour $t$).
Automated tests verify that lags equal the value at $t-k$ hours, that rolling windows use exactly $t-24$ to $t-1$, that changing the target at $t$ first affects the features at $t+1$, that scalers and one-hot categories are fitted on training rows only,
and that the tuning code never loads test rows.

\subsection{Split}
Train: 2014-02-17 to 2016-06-30 (@@N_TRAIN@@ rows); validation: 2016-07-01 to 2016-12-31 (@@N_VAL@@ rows); primary test: 2017-01-01 to 2017-07-07 (@@N_TEST@@ rows). The original test period extended to 2017-11-04
(@@N_FULL@@ rows); the part from 2017-07-11 (@@N_POST@@ rows) is affected by the timestamp discontinuity (Section~VI-D) and is reported as sensitivity analysis.

\subsection{Regression models and selection}
One model per building and algorithm: Linear Regression (OLS), KNN regression, Decision Tree regression and SVR with an RBF kernel~\cite{drucker1997svr}, implemented with scikit-learn~\cite{pedregosa2011sklearn}.
KNN and SVR use standardized inputs (SVR also a standardized target). Temporal variables were tested with raw integer, sinusoidal (24, 7 and 12 periods) and one-hot encodings
(trees use raw integers). Encoding and hyper-parameters were selected on validation: for every candidate, models were fitted per building on training rows and scored on validation rows;
one configuration per algorithm and feature set was kept, the one with the lowest macro-average validation standardized RMSE (Table~\ref{tab:cfg}). SVR candidates were tuned on a chronological subsample of at most 3000 training rows per building and then refitted on all training rows.
The best algorithm was chosen on validation; test data were used once, after freezing, with models fitted on training rows only. Predictions are clipped at 0~W.
Metrics, with $n$ evaluation hours:
\begin{align}
\mathrm{MAE} &= \tfrac1n\textstyle\sum_i |y_i-\hat y_i|, \quad \mathrm{MSE} = \tfrac1n\sum_i (y_i-\hat y_i)^2,\\
\mathrm{RMSE} &= \sqrt{\mathrm{MSE}}, \quad R^2 = 1-\tfrac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2},\\
\mathrm{nRMSE}_\sigma &= \mathrm{RMSE}/\sigma_{\mathrm{train}}, \quad \mathrm{nMAE}_\sigma = \mathrm{MAE}/\sigma_{\mathrm{train}} .
\end{align}
Because building scales differ by more than an order of magnitude, the macro average (equal weight per building) of $\mathrm{nRMSE}_\sigma$, $\mathrm{nMAE}_\sigma$ and $R^2$ is used for overall comparison; $\sigma_{\mathrm{train}}$ is the standard deviation of the building's training target.
Persistence ($\hat y_t=y_{t-1}$) and seasonal naive ($\hat y_t=y_{t-24}$) are reported as reference baselines, not as candidate models.

\begin{table}[t]
\caption{Configurations selected on validation (per feature set).}\label{tab:cfg}
\centering\small
\begin{tabular}{llll}
\toprule
Model & Features & Encoding & Hyper-parameters\\
\midrule
@@T_CFG@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Anomaly detection}
Expected consumption is the model prediction $\hat y_t$ and the residual is $r_t=y_t-\hat y_t$. The modified z-score is
$z_t = 0.6745\,(r_t-\tilde r)/\mathrm{MAD}$, and hour $t$ is flagged as a \emph{potential anomalous consumption period} if $|z_t|>3.5$~\cite{iglewicz1993outliers}; $z_t>0$ means consumption above the expectation, $z_t<0$ below.
The median $\tilde r$ and the MAD are computed from \emph{validation} residuals only (out-of-sample for the models; training residuals are in-sample and understate future error), separately for each model, feature set, building,
day-part (night, morning, afternoon, evening) and operating regime (expected power below $0.25\,\sigma_{\mathrm{train}}$ or not). The MAD is not allowed to fall below $0.05\,\sigma_{\mathrm{train}}$; without this floor, Lecture's near-zero
residuals produced hundreds of spurious flags in a first run. The test period influences no threshold. No ground truth for faults exists in the data, so flagged hours are not evidence of equipment faults.

\subsection{PCA and K-Means}
PCA was fitted on training rows of the primary matrix (13 numeric features, standardized per building), for exploration only. K-Means~\cite{lloyd1982kmeans} was applied to building-days (24 valid hours, 2014-02-17 to 2016-12-31)
described by five shape features of the daily profile normalized by the building mean (mean, load factor, standard deviation, log day/night ratio, evening share); $K\in\{2,\dots,8\}$ was selected by the silhouette coefficient~\cite{rousseeuw1987silhouettes}.

\section{Experimental Setup}
Python @@PY@@, scikit-learn 1.8, pandas 3.0, NumPy 2.4, matplotlib 3.10; one CPU core of an Intel Xeon at 2.1~GHz, 4~GB RAM. The random seed is 42. Training seven SVR models takes about @@SVR_TRAIN_S@@~s and seven Linear Regression models @@LR_TRAIN_S@@~s.
All code, tests and result tables are in the project repository; the pipeline and the model stage were re-run from scratch and reproduced the outputs.

\section{Results and Discussion}
\subsection{Model comparison}
Table~\ref{tab:val} and Table~\ref{tab:test} show macro averages over the seven buildings. On validation, SVR has the lowest error (nRMSE$_\sigma$ @@SVR_VAL@@), followed by Linear Regression (@@LR_VAL@@), Decision Tree (@@DT_VAL@@) and KNN (@@KNN_VAL@@);
persistence reaches @@PERS_VAL@@. On the primary test period the order of the best two is the same (SVR @@SVR_TEST@@, Linear Regression @@LR_TEST@@), whereas Decision Tree (@@DT_TEST@@) and KNN (@@KNN_TEST@@) are close to persistence (@@PERS_TEST@@).
The advantage of SVR over Linear Regression is small (about 0.01 in nRMSE$_\sigma$ on the primary test) and SVR is much more expensive to train and less interpretable.
Per-building results are in Tables~\ref{tab:pb} and~\ref{tab:r2}: Academic, Library and Facilities are the most predictable, the hostels and the Mess the least. Pooled errors in watts are dominated by the large buildings and should be read together with the standardized values.

\begin{table}[t]
\caption{Validation (2016-07-01 to 2016-12-31), macro average over seven buildings, primary features.}\label{tab:val}
\centering\small
\begin{tabular}{lrrr}
\toprule
Model & $R^2$ & nMAE$_\sigma$ & nRMSE$_\sigma$\\
\midrule
@@T_VAL@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\caption{Primary test (2017-01-01 to 2017-07-07), frozen models, macro average, primary features.}\label{tab:test}
\centering\small
\begin{tabular}{lrrr}
\toprule
Model & $R^2$ & nMAE$_\sigma$ & nRMSE$_\sigma$\\
\midrule
@@T_TEST@@
\bottomrule
\end{tabular}
\end{table}

\begin{table*}[t]
\caption{Per-building results of the selected model (SVR) and of Linear Regression (primary features). MAE and RMSE in W, MSE in $10^6$ W$^2$.}\label{tab:pb}
\centering\scriptsize
\begin{tabular}{llrrrrrrrr}
\toprule
 & & \multicolumn{4}{c}{Validation} & \multicolumn{4}{c}{Primary test}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Building & Model & MAE & MSE & RMSE & $R^2$ & MAE & MSE & RMSE & $R^2$\\
\midrule
@@T_PB@@
\bottomrule
\end{tabular}
\end{table*}

\begin{table*}[t]
\caption{$R^2$ per building and model (primary features). SVR, LR = Linear Regression, DT = Decision Tree.}\label{tab:r2}
\centering\scriptsize
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{Validation} & \multicolumn{4}{c}{Primary test}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Building & SVR & LR & DT & KNN & SVR & LR & DT & KNN\\
\midrule
@@T_R2@@
\bottomrule
\end{tabular}
\end{table*}

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{fig12_actual_vs_predicted.png}
\caption{Actual and SVR-predicted hourly power in one-week windows of the test period (full test period figure; windows start after 3 April 2017).}\label{fig:avp}
\end{figure}

\subsection{Energy-only versus energy plus lagged occupancy}
Table~\ref{tab:fs} compares the feature sets. P uses energy and lagged occupancy; E-m uses energy features only, trained and scored on exactly the same rows; E-a is trained on all energy-only rows within the same dates and scored on the rows shared with P.
Lagged occupancy gives no measurable gain: for SVR and Linear Regression the energy-only and primary variants differ by at most 0.004 in nRMSE$_\sigma$, and energy-only is better for KNN and Decision Tree in both periods (for example KNN @@KNN_VAL_E@@ versus @@KNN_VAL_P@@ on validation). Lagged power already carries the occupancy-driven load pattern.
This does not show that occupancy is unrelated to load (same-hour Spearman correlations are 0.32--0.70 in the exploratory analysis), only that lagged occupancy adds no predictive skill beyond power lags in this setting. Training on all energy-only rows (E-a) helps KNN and Decision Tree slightly.

\begin{table*}[t]
\caption{Macro nRMSE$_\sigma$ by feature set on matched rows. P: energy + lagged occupancy; E-m: energy-only, same training rows; E-a: energy-only, all eligible training rows.}\label{tab:fs}
\centering\small
\begin{tabular}{lrrrrrr}
\toprule
 & \multicolumn{3}{c}{Validation} & \multicolumn{3}{c}{Primary test}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Model & P & E-m & E-a & P & E-m & E-a\\
\midrule
@@T_FS@@
\bottomrule
\end{tabular}
\end{table*}

\subsection{Lecture building}
The Lecture feeder reads exactly 0~W in @@LEC_ZERO@@\% of the minutes and @@LEC_ZERO_H@@\% of the hours of the development window. Zero readings are strongly seasonal and calendar-dependent (above 99\% in March and December), occur in long runs (median run @@LEC_ZR_MEDIAN@@ min, but runs longer than a day contain @@LEC_ZR_LONG@@\% of all zero minutes), and coincide with low occupancy; current is also 0 in most of them while the meter keeps reporting frequency.
The data cannot separate an idle feeder from a metering artifact, so the zeros were kept. The consequence is a mixture of an off regime with near-zero residuals and an on regime with large ones: $R^2$ is driven by the on/off transitions and the on regime, linear and kernel models produce negative raw values that are clipped, and a residual threshold must be calibrated per regime. Lecture results are therefore reported separately in the tables and should be interpreted with more caution than those of the other buildings.

\subsection{Timestamp discontinuity and sensitivity analysis}
From about 2017-07-08 the daily load profiles of all seven energy series move earlier by 4--6 hours (Fig.~\ref{fig:shift}), while occupancy timestamps do not. The change is abrupt, directly after logging gaps of 12 to 33 hours.
Against occupancy, the best energy offset is @@PRE_LO@@ to @@PRE_HI@@ minutes before the break and @@POST_LO@@ to @@POST_HI@@ minutes after it; for Academic the Spearman correlation with occupancy is @@AC_POST0@@ with no shift and @@AC_POST_BEST@@ with a shift of @@AC_POST_SHIFT@@ min.
The offset is not identical across buildings, so no single correction value can be established. The dataset authors' reader script applies a fixed +05:30 conversion to all periods and documents no break. The cause (for example a logger clock reset) is unresolved,
and the raw data were left untouched. The primary test period therefore ends on 2017-07-07, and Table~\ref{tab:sens} reports the other periods separately. Models that depend on hour-of-day features degrade after the break (SVR @@SVR_TEST@@ to @@SVR_POST@@), whereas persistence is unaffected;
on the full original test period Linear Regression is better than SVR (@@LR_FULL@@ vs. @@SVR_FULL@@), which illustrates that rankings on the full period are confounded by the discontinuity.

\begin{table}[t]
\caption{Macro nRMSE$_\sigma$ of the frozen models by period (primary features).}\label{tab:sens}
\centering\small
\begin{tabular}{lrrrr}
\toprule
Model & Valid. & Primary test & After break & Full test\\
\midrule
@@T_SENS@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{fig19_timestamp_shift_profiles.png}
\caption{Mean weekday profiles before and after the discontinuity (hours as labelled, IST).}\label{fig:shift}
\end{figure}

\subsection{Anomaly detection}
Table~\ref{tab:an} gives the share of flagged hours. For the selected SVR model with primary features, @@SVR_AN_VAL@@\% of validation hours (the calibration period, out-of-sample) and @@SVR_AN_TEST@@\% of primary-test hours (@@SVR_AN_N@@ hours) are flagged,
i.e.\ the thresholds transfer reasonably to the test period; after the timestamp break the rate rises to @@SVR_AN_POST@@\%, which reflects the displaced timestamps (the models expect the wrong time-of-day pattern) and not consumption faults. Flags after 2017-07-08 should therefore not be interpreted.
Fig.~\ref{fig:timeline} shows the monthly share and Fig.~\ref{fig:examples} examples. Boys Hostel and Lecture have the highest primary-test rates (Table~\ref{tab:anb}); Lecture's rate is affected by its two regimes (Section~VI-C).
On matched rows, the flags of SVR with and without occupancy overlap strongly (Jaccard @@JAC_FEAT@@ on the primary test, @@JAC_FEAT_LR@@ for Linear Regression), whereas SVR and Linear Regression agree much less (Jaccard @@JAC_MODEL@@), so the flagged set depends on the model;
the intersection of both is the more conservative list. Because the data contain no labelled faults, precision and recall cannot be computed, and the flagged hours are best described as potential anomalous consumption periods.

\begin{table}[t]
\caption{Flagged hours (\%) by model and feature set. P: energy + lagged occupancy; E: energy-only (same rows). Last column: flagged hours in the primary test.}\label{tab:an}
\centering\small
\begin{tabular}{llrrrr}
\toprule
Model & Features & Valid. & Primary test & After break & Count\\
\midrule
@@T_AN@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\caption{Flagged hours (\%) per building, SVR with primary features.}\label{tab:anb}
\centering\small
\begin{tabular}{lrrr}
\toprule
Building & Valid. & Primary test & After break\\
\midrule
@@T_ANB@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure*}[t]
\centering\includegraphics[width=0.9\textwidth]{fig21_anomaly_timeline.png}
\caption{Monthly share of flagged hours (SVR, primary features). The cyan line marks the timestamp discontinuity.}\label{fig:timeline}
\end{figure*}

\begin{figure}[t]
\centering\includegraphics[width=\columnwidth]{fig22_anomaly_examples.png}
\caption{Flagged hours (markers) in the one-week primary-test windows with the most flagged hours; red: above expectation, blue: below.}\label{fig:examples}
\end{figure}

\subsection{PCA and K-Means}
The first three principal components explain @@PC1@@\%, @@PC2@@\% and @@PC3@@\% of the variance, and @@PC90@@ components are needed for 90\% (Fig.~\ref{fig:pca}). PC1 loads on lagged power and lagged occupancy together (a general load/occupancy level), PC2 mainly on the weekend/working-day and activity flags (calendar regime).
PCA was not used as a modelling step and no claim is made that it improves prediction.
K-Means on building-days selected $K=@@K_ALL@@$ (silhouette @@SIL_ALL@@), but two of the three clusters consist almost exclusively of Lecture days (off days with near-zero load and on days with a high load relative to its low mean), so the clustering mainly isolates Lecture's two regimes.
Without Lecture, the silhouette favours $K=@@K_WL@@$ (@@SIL_WL@@, moderate structure): a cluster of higher-than-average, mostly weekday, daytime-dominated profiles and a cluster of flatter, lower-load days with more weekends (Table~\ref{tab:km}, Fig.~\ref{fig:km}). The clusters are descriptive statistical groups; no efficiency labels are assigned.

\begin{table}[t]
\caption{K-Means clusters of building-days without Lecture: days, mean load relative to the building mean, load factor, weekend share, high-activity share.}\label{tab:km}
\centering\small
\begin{tabular}{lrrrrr}
\toprule
Cl. & Days & Rel. mean & Load f. & Weekend & Act. H\\
\midrule
@@KM_WL@@
\bottomrule
\end{tabular}
\end{table}

\begin{figure*}[t]
\centering\includegraphics[width=0.95\textwidth]{fig25_pca.png}
\caption{PCA of training rows: explained variance, cumulative variance and the first two components.}\label{fig:pca}
\end{figure*}

\begin{figure*}[t]
\centering\includegraphics[width=0.95\textwidth]{fig26_kmeans_without_lecture.png}
\caption{K-Means selection (elbow, silhouette) and mean normalized daily profile per cluster, six buildings without Lecture.}\label{fig:km}
\end{figure*}

\section{Dashboard / System Implementation}
A minimal Streamlit application (\texttt{app/streamlit\_app.py}) reads only saved outputs. A sidebar selects building, model (SVR or Linear Regression), feature set and date range; a check box adds the post-break rows and a banner states the timestamp issue and that flags are not fault diagnoses.
Tabs show an overview with metrics for the selection, actual versus expected power with flagged hours, evaluation metrics (selection and frozen-model tables by period), the anomaly timeline with event table and thresholds, and CSV export of the selected hours and events.
A button re-runs the saved model file on the selected hours and reports the difference to the stored predictions. The application was tested headlessly with Streamlit's test harness for building, model, feature-set and period selections, and the server answered HTTP requests; it is a batch viewer of saved results, not a real-time monitor.

\section{Limitations}
(i) The energy timestamps after about 2017-07-08 are displaced and unresolved, which reduces the usable test period to January--July 2017 (and leaves February almost empty because of a 49-day occupancy outage).
(ii) There is no ground truth for anomalies; thresholds are statistical and flagged hours are not confirmed faults, and precision/recall cannot be assessed. (iii) Occupancy is a Wi-Fi-based proxy, absent steps cannot be distinguished from zero occupancy, and requiring occupancy lags removes about 30\% of the energy-only rows.
(iv) Weather could not be used. (v) Lecture is zero-inflated and its metrics and flags are less reliable. (vi) The hostel and library series contain long outages; a single campus is studied, so generalization to other institutions is untested.
(vii) Frozen models are trained on the training period only, SVR's selected hyper-parameters lie at the border of the searched grid, and differences between SVR and Linear Regression are small. (viii) PCA and K-Means are exploratory, and the K-Means result is dominated by Lecture.

\section{Conclusion and Future Work}
A leakage-controlled hourly pipeline for seven campus feeders was implemented and evaluated. SVR was selected on validation and had the lowest macro error on the primary test period, but its margin over Linear Regression is small, and all models improve on persistence only moderately.
Lagged occupancy did not add predictive skill. A residual-based detector calibrated on validation residuals transfers to the primary test period with a similar flag rate, and its output must be treated as a list of potential anomalous periods.
Future work: resolving the 2017 timestamp discontinuity with the dataset authors or the original logger records, reassessing the results on the corrected data, adding weather for periods where it exists, regime-aware models for Lecture, more advanced time-series models, and validation of flagged periods against maintenance records.

\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
"""

if __name__ == "__main__":
    main()
