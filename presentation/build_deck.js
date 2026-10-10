// Builds campus_energy_presentation.pptx from deck_data.json (numbers) and results/figures (images).
// Usage (from presentation/): node build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");
let applyTheme = async () => {};            // optional: writes the theme colours into the deck (script ships with the pptx skill)
try { ({ applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js")); } catch (e) { console.log("apply_theme.js not found; deck keeps the default theme colours"); }

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const FIG = path.join(__dirname, "..", "results", "figures");
const f3 = (x) => x.toFixed(3);
const n0 = (x) => x.toLocaleString("en-US");

const THEME = {
  name: "CampusEnergy", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1F2937", lt1: "FFFFFF", dk2: "1F3A5F", lt2: "F3F4F6", accent1: "0072B2", accent2: "D55E00", accent3: "009E73",
            accent4: "E69F00", accent5: "CC79A7", accent6: "6B7280", hlink: "0072B2", folHlink: "6B7280" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Smart Campus Energy Consumption Prediction and Anomaly Detection";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_ONLY", background: { color: "FFFFFF" },
  objects: [
    { rect: { x: 0, y: 0, w: 10, h: 0.08, fill: { color: "1F3A5F" } } },
    { line: { x: 0.45, y: 0.95, w: 9.1, h: 0, line: { color: "D1D5DB", width: 0.75 } } },
  ],
  slideNumber: { x: 9.2, y: 5.28, w: 0.5, h: 0.25, fontSize: 9, color: "6B7280", align: "right" },
  margin: [0.3, 0.45, 0.4, 0.45],
});
pres.defineSlideMaster({ title: "TITLE_SLIDE", background: { color: "1F3A5F" }, objects: [], margin: [0.3, 0.6, 0.3, 0.6] });

let SECTION = "Deck";
function slide(title, notes) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: SECTION });
  s.addText(title, { x: 0.45, y: 0.18, w: 9.1, h: 0.72, fontSize: 21, bold: true, color: "1F3A5F", valign: "middle", margin: 0, isTextBox: true, fit: "shrink" });
  if (notes) s.addNotes(notes);
  return s;
}
function bullets(s, items, o) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 5 } }));
  s.addText(runs, Object.assign({ isTextBox: true, fontSize: 12, color: "1F2937", valign: "top", margin: 0.04 }, o));
}
function fit(s, file, x, y, w, h, name) {
  const [iw, ih] = D.fig_sizes[file]; const r = Math.min(w / iw, h / ih);
  const ww = iw * r, hh = ih * r;
  s.addImage({ path: path.join(FIG, file), x: x + (w - ww) / 2, y: y + (h - hh) / 2, w: ww, h: hh, altText: name || file });
}
const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "1F3A5F" }, fontSize: 10, align: "center" } });
const cell = (t, o) => ({ text: String(t), options: Object.assign({ fontSize: 10, color: "1F2937" }, o || {}) });

// 1 ---------------------------------------------------------------- title
SECTION = "Introduction";
pres.addSection({ title: SECTION });
let s = pres.addSlide({ masterName: "TITLE_SLIDE", sectionTitle: SECTION });
s.addText("Machine Learning-Based Smart Campus Energy Consumption Prediction and Anomaly Detection",
  { x: 0.6, y: 1.1, w: 8.8, h: 1.5, fontSize: 30, bold: true, color: "FFFFFF", isTextBox: true, margin: 0, valign: "top" });
s.addText("Jahnavi Kallolini  |  Navya Menon  |  Keshav Agrawal", { x: 0.6, y: 3.0, w: 8.8, h: 0.4, fontSize: 16, color: "FFFFFF", isTextBox: true, margin: 0 });
s.addText("B.Tech CSE (AI&ML), School of Computer Engineering, Manipal Institute of Technology\nMentors: Dr. Narendra V. G., Dr. Ashwath Rao",
  { x: 0.6, y: 3.5, w: 8.8, h: 0.8, fontSize: 12, color: "D1D5DB", isTextBox: true, margin: 0 });
s.addNotes("Mini-project based on the submitted synopsis. All numbers in this deck come from the project's saved result tables.");

// 2 ---------------------------------------------------------------- problem + data
s = slide("Problem, data and objectives", "Objectives follow the synopsis. The weather file covers only March-June 2018, after the last energy reading, so it cannot be used; backup feeders are excluded to keep seven buildings.");
s.addText("Objectives", { x: 0.45, y: 1.1, w: 4.4, h: 0.3, fontSize: 14, bold: true, color: "0072B2", isTextBox: true, margin: 0 });
bullets(s, ["Predict hourly power of seven campus building feeders", "Compare Linear Regression, KNN, Decision Tree and SVR", "Flag potential anomalous consumption from prediction residuals", "Explore patterns with PCA and K-Means"], { x: 0.45, y: 1.5, w: 4.4, h: 2.6, fontSize: 14 });
s.addText("Data (I-BLEND)", { x: 5.2, y: 1.1, w: 4.3, h: 0.3, fontSize: 14, bold: true, color: "0072B2", isTextBox: true, margin: 0 });
bullets(s, ["Aug 2013 to Dec 2017; seven main feeders", "Power every minute (W); Wi-Fi occupancy every 10 min; daily academic calendar", "Target: hourly mean power", "Weather excluded: only Mar-Jun 2018, no overlap", "Hostel UPS (backup) feeders excluded"], { x: 5.2, y: 1.5, w: 4.3, h: 2.8, fontSize: 13 });
s.addText("Buildings: Academic, Boys Hostel, Facilities, Girls Hostel, Lecture, Library, Mess", { x: 0.45, y: 4.2, w: 9.1, h: 0.4, fontSize: 11, color: "6B7280", isTextBox: true, margin: 0 });

// 3 ---------------------------------------------------------------- preprocessing
s = slide("Preprocessing and features", "Missing occupancy stays missing: the released occupancy never reports zero. Short gaps in lag inputs use causal carry-forward of at most one hour; targets are never filled.");
bullets(s, ["Hour valid if at least 45/60 power readings (occupancy: 4/6)", "Long outages kept missing (hostels and library: 200+ days); never filled", "Each building modelled on its own valid data", "Features: hour, weekday, weekend, month, working day, activity H/L; power lags 1, 2, 24; 24 h rolling mean/std; occupancy lags 1, 2, 24", "No current-hour occupancy or future information"], { x: 0.45, y: 1.15, w: 4.7, h: 3.9, fontSize: 13 });
const rows = [[hdr("Building"), hdr("Valid hours"), hdr("Primary rows")]];
Object.keys(D.labels).forEach((b) => rows.push([cell(D.labels[b]), cell(n0(D.valid_hours[b]), { align: "right" }), cell(n0(D.primary_rows[b]), { align: "right" })]));
s.addTable(rows, { x: 5.4, y: 1.15, w: 4.1, colW: [1.7, 1.2, 1.2], border: { type: "solid", color: "D1D5DB", pt: 0.5 }, rowH: 0.3 });
s.addText("Primary rows need an observed target and all lag features, including occupancy.", { x: 5.4, y: 3.7, w: 4.1, h: 0.5, fontSize: 10, color: "6B7280", isTextBox: true, margin: 0 });

// 4 ---------------------------------------------------------------- protocol
s = slide("Evaluation protocol and leakage control", "Models are fitted on training rows only and evaluated once on validation and test. The tuning code never loads test rows; automated tests check lags, rolling windows, scalers, encodings and the split.");
const prow = [[hdr("Period"), hdr("Dates"), hdr("Rows")],
  [cell("Train"), cell("2014-02-17 to 2016-06-30"), cell(n0(D.rows.train), { align: "right" })],
  [cell("Validation"), cell("2016-07-01 to 2016-12-31"), cell(n0(D.rows.val), { align: "right" })],
  [cell("Primary test", { bold: true }), cell("2017-01-01 to 2017-07-07", { bold: true }), cell(n0(D.rows.test_primary), { align: "right", bold: true })],
  [cell("Sensitivity (after break)", { color: "D55E00" }), cell("2017-07-11 to 2017-11-04", { color: "D55E00" }), cell(n0(D.rows.post), { align: "right", color: "D55E00" })]];
s.addTable(prow, { x: 0.45, y: 1.15, w: 5.2, colW: [1.8, 2.4, 1.0], border: { type: "solid", color: "D1D5DB", pt: 0.5 }, rowH: 0.36 });
bullets(s, ["Chronological split, no shuffling", "Encoding and hyper-parameters chosen on validation only", "One model per building, same procedure for all", "Scalers and one-hot categories fitted on training rows", "Metrics: MAE, MSE, RMSE, R2 per building; nRMSE = RMSE / training std for comparison across scales", "Persistence and seasonal-naive baselines for reference"], { x: 5.9, y: 1.1, w: 3.7, h: 4.0, fontSize: 12.5 });
s.addText("The primary test period ends before a timestamp discontinuity (slide 7).", { x: 0.45, y: 3.0, w: 5.2, h: 0.5, fontSize: 11, color: "D55E00", isTextBox: true, margin: 0 });

// 5 ---------------------------------------------------------------- model comparison (native chart)
s = slide("Model comparison: SVR lowest error, margin over Linear Regression small", "Macro average over seven buildings of RMSE divided by each building's training standard deviation (lower is better). SVR was selected on validation. On the primary test period it is still best; after the break all time-of-day dependent models degrade.");
const cats = ["SVR", "Linear", "Decision Tree", "KNN", "Persistence"];
const series = [["Validation", "validation"], ["Primary test", "test_primary"], ["After break (sensitivity)", "test_post_break"]];
s.addChart(pres.charts.BAR, series.map(([n, k]) => ({ name: n, labels: cats, values: cats.map((c) => +D.macro[k][c].toFixed(3)) })), {
  x: 0.45, y: 1.05, w: 5.9, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: ["0072B2", "009E73", "D55E00"],
  showTitle: true, title: "Macro nRMSE (lower is better)", titleFontSize: 11, titleColor: "1F2937",
  showValue: true, dataLabelFontSize: 8, dataLabelFormatCode: "0.00", dataLabelPosition: "outEnd",
  catAxisLabelColor: "1F2937", valAxisLabelColor: "6B7280", valAxisMinVal: 0, valAxisMaxVal: 0.7, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 9, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
});
bullets(s, [`Validation R2: SVR ${f3(D.macroR2.validation.SVR)}, Linear ${f3(D.macroR2.validation.Linear)}`, `Primary test R2: SVR ${f3(D.macroR2.test_primary.SVR)}, Linear ${f3(D.macroR2.test_primary.Linear)}`, "Decision Tree and KNN are close to persistence on test", `SVR training ${D.train_seconds.SVR.toFixed(0)} s vs ${D.train_seconds.Linear.toFixed(1)} s for Linear (7 buildings)`, "Full original test period: Linear Regression ranks first (confounded by the break)"], { x: 6.55, y: 1.1, w: 3.0, h: 4.0, fontSize: 12 });

// 6 ---------------------------------------------------------------- per building + feature sets
s = slide("Per building results and role of occupancy", "R2 of the selected SVR and Linear Regression per building. Lagged occupancy gave no measurable gain on identical rows; Lecture reads exactly 0 W in most hours.");
const t2 = [[hdr("R2"), hdr("SVR val"), hdr("SVR test"), hdr("Linear val"), hdr("Linear test")]];
Object.keys(D.labels).forEach((b) => t2.push([cell(D.labels[b]), cell(f3(D.perb[b].validation.SVR), { align: "right" }), cell(f3(D.perb[b].test_primary.SVR), { align: "right" }), cell(f3(D.perb[b].validation.Linear), { align: "right" }), cell(f3(D.perb[b].test_primary.Linear), { align: "right" })]));
s.addTable(t2, { x: 0.45, y: 1.1, w: 5.2, colW: [1.4, 0.95, 0.95, 0.95, 0.95], border: { type: "solid", color: "D1D5DB", pt: 0.5 }, rowH: 0.36 });
const FS = D.fs;
bullets(s, [`Energy-only vs energy + lagged occupancy (matched rows): SVR ${f3(FS.test_primary.SVR.E_matched)} vs ${f3(FS.test_primary.SVR.P)} on test; KNN ${f3(FS.test_primary.KNN.E_matched)} vs ${f3(FS.test_primary.KNN.P)}`, "Lagged power already carries the occupancy-driven pattern", `Lecture: ${D.lecture.zero_pct.toFixed(1)}% of minutes at exactly 0 W (${D.lecture.hourly_zero_pct.toFixed(1)}% of hours), seasonal and long runs; zeros kept, not removed`, "Lecture mixes an off and an on regime: R2 is driven by on/off transitions; results reported separately"], { x: 5.9, y: 1.1, w: 3.7, h: 4.0, fontSize: 12.5 });

// 7 ---------------------------------------------------------------- timestamp discontinuity
s = slide("Timestamp discontinuity from July 2017 (unresolved)", "Daily profiles of all seven energy series move earlier by 4-6 h from about 2017-07-08, right after logging gaps; occupancy does not shift. The authors' reader script applies a fixed +05:30 conversion and documents no break. Nothing was corrected.");
fit(s, "fig19_timestamp_shift_profiles.png", 0.45, 1.05, 9.1, 2.55, "Weekday profiles before and after the break");
bullets(s, ["Monthly profile alignment: about 0 h until June 2017, 4-6 h in July-December 2017, all buildings", "Against occupancy the best energy offset is 0-70 min before and 250-330 min after the break; not identical across buildings, so no exact correction can be justified", "Raw data untouched. Primary test ends 2017-07-07; later rows reported only as sensitivity", "SVR macro nRMSE rises from 0.382 (before) to 0.503 (after the break)"], { x: 0.45, y: 3.7, w: 9.1, h: 1.6, fontSize: 11.5 });

// 8 ---------------------------------------------------------------- anomaly detection
s = slide("Residual-based anomaly detection", "Flagged hours are statistical outliers relative to the model; there is no ground truth for faults. Thresholds use validation residuals only.");
bullets(s, ["Residual = actual - expected (SVR, primary features)", "Flag if |modified z| > 3.5; median and MAD from validation residuals per building, day-part and regime", `Validation ${D.anomaly.validation.SVR_P.toFixed(1)}% flagged; primary test ${D.anomaly.test_primary.SVR_P.toFixed(1)}% (${n0(D.anomaly_test_count)} hours); after the break ${D.anomaly.test_post_break.SVR_P.toFixed(1)}% (timestamp effect)`, `With vs without occupancy: flags overlap strongly (Jaccard ${D.jaccard["features: SVR P vs SVR E_matched"].toFixed(2)}); SVR vs Linear overlap less (${D.jaccard["models: SVR P vs LR P"].toFixed(2)})`, "Output = potential anomalous consumption periods, not diagnosed faults"], { x: 0.45, y: 1.1, w: 3.7, h: 4.1, fontSize: 12 });
fit(s, "fig21_anomaly_timeline.png", 4.3, 1.4, 5.3, 3.2, "Monthly share of flagged hours");

// 9 ---------------------------------------------------------------- PCA + KMeans
s = slide("PCA and K-Means (exploratory)", "PCA is not used as a modelling step. K-Means on all seven buildings is dominated by Lecture's two regimes; without Lecture the structure is moderate.");
fit(s, "fig25_pca.png", 0.45, 1.05, 6.1, 1.9, "PCA");
fit(s, "fig26_kmeans_without_lecture.png", 0.45, 3.05, 6.1, 1.9, "K-Means without Lecture");
bullets(s, [`PC1-PC3 explain ${(D.pca.pc1 * 100).toFixed(1)}%, ${(D.pca.pc2 * 100).toFixed(1)}%, ${(D.pca.pc3 * 100).toFixed(1)}%; ${D.pca.components_for_90pct} components for 90%`, "PC1: lagged power and occupancy level; PC2: calendar regime", `K-Means, seven buildings: K=${D.kmeans_all7.selected_K} (silhouette ${D.kmeans_all7.silhouette_at_K.toFixed(2)}), two of three clusters are Lecture-only`, `Without Lecture: K=${D.kmeans_wl.selected_K} (silhouette ${D.kmeans_wl.silhouette_at_K.toFixed(2)}): weekday daytime-heavy days vs flatter low-load days`, "Clusters are descriptive; no efficiency labels"], { x: 6.7, y: 1.05, w: 2.9, h: 4.1, fontSize: 10.5 });

// 10 --------------------------------------------------------------- dashboard, limits, conclusion
SECTION = "Conclusion";
pres.addSection({ title: SECTION });
s = slide("Dashboard, limitations and conclusion", "Dashboard run: streamlit run app/streamlit_app.py. It reads saved outputs only. The deck has no dashboard screenshot; the dashboard was tested headlessly.");
const col = (x, title, items, color) => { s.addText(title, { x, y: 1.1, w: 2.9, h: 0.3, fontSize: 13, bold: true, color, isTextBox: true, margin: 0 }); bullets(s, items, { x, y: 1.45, w: 2.9, h: 3.7, fontSize: 11.5 }); };
col(0.45, "Dashboard (Streamlit)", ["Filters: building, model, feature set, dates", "Actual vs expected power, metrics, anomaly timeline and events", "CSV export of selection and events", "Banner states timestamp issue and that flags are not faults", "Batch viewer of saved results, not real-time"], "0072B2");
col(3.55, "Limitations", ["Timestamps from July 2017 displaced (unresolved)", "No ground truth for anomalies", "Occupancy is a Wi-Fi proxy; absent values not read as zero", "Lecture is zero-inflated; weather unavailable", "SVR vs Linear margin small; SVR grid edge", "One campus only"], "D55E00");
col(6.65, "Conclusion and next steps", ["SVR selected on validation; best on primary test", "Lagged occupancy adds no predictive skill here", "Anomaly flags transfer to test with similar rate", "Next: resolve the timestamp issue with the data authors, re-evaluate, regime-aware Lecture model, validate flags against maintenance records"], "009E73");

pres.writeFile({ fileName: path.join(__dirname, "campus_energy_presentation.pptx") }).then(async (f) => {
  await applyTheme(f, THEME);
  console.log("written", f);
});
