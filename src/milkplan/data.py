"""Fetch, parse and validate Eurostat monthly milk collection (apro_mk_colm) for one country."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/apro_mk_colm?format=JSON&lang=EN&geo={geo}"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def raw_path(geo: str) -> Path:
    return RAW_DIR / f"eurostat_apro_mk_colm_{geo}.json"


def fetch(geo: str = "DK") -> Path:
    """Download a fresh snapshot. The committed snapshot is used when running offline."""
    import urllib.request
    path = raw_path(geo)
    with urllib.request.urlopen(URL.format(geo=geo), timeout=120) as r:
        path.write_bytes(r.read())
    return path


def parse(doc: dict, dairyprod: str, milkitem: str, unit: str) -> pd.Series:
    """Turn a JSON-stat document into a monthly series for one product/item/unit."""
    dims = doc["id"]
    sizes = doc["size"]
    index = {d: doc["dimension"][d]["category"]["index"] for d in dims}
    want = {"freq": "M", "dairyprod": dairyprod, "milkitem": milkitem, "unit": unit}
    fixed = {}
    for d in dims:
        if d == "time":
            continue
        if d in want:
            if want[d] not in index[d]:
                raise KeyError(f"{d}={want[d]} not in data")
            fixed[d] = index[d][want[d]]
        else:  # single-valued dimension such as geo
            fixed[d] = 0
    strides = np.cumprod([1] + sizes[::-1][:-1])[::-1]
    out = {}
    for period, t in index["time"].items():
        pos = sum((t if d == "time" else fixed[d]) * s for d, s in zip(dims, strides))
        val = doc["value"].get(str(pos))
        if val is not None:
            out[pd.Period(period, "M")] = float(val)
    s = pd.Series(out).sort_index()
    s.index = s.index.to_timestamp()
    s.index.freq = "MS"
    return s


def load(geo: str = "DK") -> pd.DataFrame:
    doc = json.loads(raw_path(geo).read_text())
    df = pd.DataFrame({
        "milk_kt": parse(doc, "D1110D", "PRD", "THS_T"),
        "fat_pct": parse(doc, "D1110D", "FAT", "PC") if _has(doc, "D1110D", "FAT", "PC") else np.nan,
    })
    return df.asfreq("MS")


def _has(doc, dairyprod, milkitem, unit) -> bool:
    try:
        return len(parse(doc, dairyprod, milkitem, unit)) > 0
    except KeyError:
        return False


def validate(s: pd.Series, start: str = "2000-01-01") -> dict:
    """Rules checked before any model sees the data. Gaps are reported, never filled silently."""
    s = s[s.index >= start]
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    missing = full.difference(s.dropna().index)
    # Outlier: more than 25 % away from the same month last year
    yoy = (s / s.shift(12) - 1).abs()
    return {
        "first": str(s.index.min().date()),
        "last": str(s.index.max().date()),
        "months": int(s.notna().sum()),
        "missing_months": [str(m.date()) for m in missing],
        "negative_values": int((s < 0).sum()),
        "yoy_jumps_over_25pct": [str(i.date()) for i in yoy[yoy > 0.25].index],
    }
