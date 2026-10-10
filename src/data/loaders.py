"""Raw-file readers. Raw files are never modified; everything is read-only."""
import numpy as np
import pandas as pd

from src import config as C


def energy_path(building: str):
    return C.ENERGY_DIR / C.BUILDINGS[building]["energy"]


def load_energy_minutely(building: str, usecols=("timestamp", "power")) -> pd.DataFrame:
    """Minute-level energy file with integer epoch seconds in column `timestamp`."""
    df = pd.read_csv(energy_path(building), usecols=list(usecols), na_values=["NA"])
    ts = df["timestamp"]
    if not np.all(ts.dropna() % 1 == 0):
        raise ValueError(f"{building}: non-integer timestamps found")
    df["timestamp"] = ts.astype("int64")
    return df


def load_occupancy(building: str) -> pd.DataFrame:
    code = C.BUILDINGS[building]["occ"]
    df = pd.read_csv(C.OCC_DIR / f"{code}.csv")
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def load_calendar() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(C.CAL_DIR.glob("calender_year_*_.csv"))]
    cal = pd.concat(parts, ignore_index=True)
    cal["date"] = pd.to_datetime(cal["Date"])
    cal = cal.drop(columns="Date").sort_values("date").reset_index(drop=True)
    if cal["date"].duplicated().any():
        raise ValueError("duplicate dates in calendar files")
    return cal


def load_weather() -> pd.DataFrame:
    w = pd.read_csv(C.WEATHER_DIR / "IIITD_and_airport_data.csv", index_col=0)
    w.index = pd.to_datetime(w.index)
    w.index.name = "timestamp"
    return w


def hour_key(epoch_seconds):
    """Local (IST) hour index = number of whole hours since 1970-01-01 00:00 IST."""
    return (np.asarray(epoch_seconds) + C.IST_OFFSET_S) // 3600


def hour_key_to_timestamp(keys) -> pd.DatetimeIndex:
    """Hour keys -> tz-naive IST wall-clock timestamps (hour start)."""
    return pd.to_datetime(np.asarray(keys, dtype="int64") * 3600, unit="s")
