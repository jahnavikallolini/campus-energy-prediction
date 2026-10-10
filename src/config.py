"""Central configuration for the I-BLEND campus energy project.

Every threshold or setting used by the pipeline lives here so that the code,
the report and the dashboard refer to one source of truth.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("IBLEND_RAW_DIR", ROOT / "data" / "raw"))
ENERGY_DIR = RAW_DIR / "energy"
OCC_DIR = RAW_DIR / "occupancy"
WEATHER_DIR = RAW_DIR / "weather"
CAL_DIR = RAW_DIR / "calendar"
PROC_DIR = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG_DIR, TAB_DIR, MET_DIR = RESULTS / "figures", RESULTS / "tables", RESULTS / "metrics"
REPORT_DIR = ROOT / "reports"

SEED = 42

# --- Time handling -------------------------------------------------------
# Unix timestamps are UTC; the dataset documentation specifies Asia/Kolkata (+05:30).
# India has no daylight saving, so a fixed offset is exact.
IST_OFFSET_S = 5 * 3600 + 30 * 60
# All processed tables use tz-naive IST wall-clock timestamps, labelled by hour start.

# --- Buildings (seven main feeders, as in the synopsis) --------------------
BUILDINGS = {
    "Academic":   dict(energy="acad_build_mains.csv",       occ="ACB", label="Academic"),
    "Boys_main":  dict(energy="boys_hostel_mains.csv",      occ="BH",  label="Boys Hostel"),
    "Facilities": dict(energy="facilities_build_mains.csv", occ="SRB", label="Facilities"),
    "Girls_main": dict(energy="girls_hostel_mains.csv",     occ="GH",  label="Girls Hostel"),
    "Lecture":    dict(energy="lecture_build_mains.csv",    occ="LCB", label="Lecture"),
    "Library":    dict(energy="library_build_mains.csv",    occ="LB",  label="Library"),
    "Mess":       dict(energy="mess_build_mains.csv",       occ="DB",  label="Mess (Dining)"),
}
# Backup (UPS) feeders exist only in all_buildings_power.csv. They are excluded from the
# primary target to preserve the seven-building scope defined in the synopsis.
EXCLUDED_FEEDERS = ["Boys_backup", "Girls_backup"]

# --- Hourly aggregation ----------------------------------------------------
MIN_VALID_ENERGY_MIN = 45      # of 60 one-minute readings required for a valid hour
MIN_VALID_OCC_OBS = 4          # of 6 ten-minute observations required for a valid hour
ENERGY_THRESHOLD_GRID = [30, 40, 45, 50, 55, 60]
OCC_THRESHOLD_GRID = [3, 4, 5, 6]

# --- Gap handling ------------------------------------------------------------
# Maximum run of consecutive missing hours bridged by causal last-observation-carried-
# forward when building lag/rolling INPUTS (never targets). Justified by the masking
# experiment in src/data/preprocess.py (see reports/preprocessing_report.md).
MAX_FILL_HOURS = 1
ROLL_WINDOW = 24
ROLL_MIN_PERIODS = 18
LAGS_POWER = [1, 2, 24]
LAGS_OCC = [1, 2, 24]

# --- Split -------------------------------------------------------------------
SPLIT_TRAIN_SHARE_RANGE = (0.55, 0.75)
SPLIT_VAL_SHARE_RANGE = (0.10, 0.20)
SPLIT_TEST_SHARE_RANGE = (0.10, 0.25)

POWER_UNIT = "W"   # confirmed in the I-BLEND Data Descriptor ("power (watts)")

# --- Test periods after the timestamp investigation (see reports/timestamp_investigation.md) -----------
# Energy timestamps appear displaced by ~4.5-5.5 h from about 2017-07-08 (unresolved, NOT corrected).
# Primary test period ends before the break; the rest of the original test period is a sensitivity analysis.
PRIMARY_TEST_START = "2017-01-01"
PRIMARY_TEST_END_EXCL = "2017-07-08"      # primary test = 2017-01-01 .. 2017-07-07
POST_BREAK_START = "2017-07-11"           # 2017-07-08 .. 07-10 is a transition and belongs to neither sub-period

# --- Anomaly detection ----------------------------------------------------------------------------
ANOMALY_SCALE_FLOOR_FRAC = 0.05           # scale floor: stratum MAD >= 0.05 x training-target std of the building
ANOMALY_LOW_REGIME_FRAC = 0.25            # expected power < 0.25 x training std => "off-like" regime stratum
ANOMALY_MIN_STRATUM_ROWS = 50
ANOMALY_Z = 3.5                           # modified z-score cut-off (Iglewicz & Hoaglin, 1993)
DAYPARTS = {"night": (0, 5), "morning": (6, 11), "afternoon": (12, 17), "evening": (18, 23)}
