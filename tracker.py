"""Retention action log: what HR did, for whom, and what happened."""
from datetime import date
from pathlib import Path

import pandas as pd

COLS = ["Date", "EmployeeNumber", "Action", "Owner", "Status", "Outcome", "RiskAtLogging", "Notes"]
STATUS = ["Planned", "In progress", "Done", "Declined"]
OUTCOME = ["Unknown", "Stayed", "Left"]


def load(path: Path) -> pd.DataFrame:
    if Path(path).exists():
        try:
            df = pd.read_csv(path)
            for c in COLS:
                if c not in df.columns:
                    df[c] = ""
            return df[COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=COLS)


def save(df: pd.DataFrame, path: Path) -> None:
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except OSError:
        pass  # read-only hosting: the download button is the backup


def add(log: pd.DataFrame, items, owner="HR") -> tuple[pd.DataFrame, int]:
    """items: iterable of (employee_number, action, risk). Skips exact duplicates still open."""
    new = []
    for emp, action, risk in items:
        dup = ((log["EmployeeNumber"] == emp) & (log["Action"] == action) &
               (log["Status"].isin(["Planned", "In progress"]))).any()
        dup = dup or any(n["EmployeeNumber"] == int(emp) and n["Action"] == action for n in new)
        if not dup:
            new.append({"Date": date.today().isoformat(), "EmployeeNumber": int(emp), "Action": action,
                        "Owner": owner, "Status": "Planned", "Outcome": "Unknown",
                        "RiskAtLogging": round(float(risk), 3), "Notes": ""})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
    return log, len(new)


def summary(log: pd.DataFrame) -> dict:
    n = len(log)
    if n == 0:
        return {"actions": 0, "done_pct": 0.0, "known": 0, "stay_rate": None, "expected_stay": None}
    known = log[log["Outcome"].isin(["Stayed", "Left"])]
    stay = (known["Outcome"] == "Stayed").mean() if len(known) else None
    exp = 1 - pd.to_numeric(known["RiskAtLogging"], errors="coerce").mean() if len(known) else None
    return {"actions": n, "done_pct": (log["Status"] == "Done").mean(), "known": len(known),
            "stay_rate": stay, "expected_stay": exp}
