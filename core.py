"""TalentGuard core: model, honest risk scores, SHAP-style drivers, recommendations, HR copilot."""
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constant / ID columns that only add noise (the "traps" in the brief)
USELESS = ["EmployeeCount", "Over18", "StandardHours"]
PROTECTED = ["Age", "Gender", "MaritalStatus"]

# driver -> (theme, action). Only things HR can actually change appear here.
ACTIONS = {
    "OverTime": ("Workload", "Cap or reduce overtime; rebalance workload or add headcount"),
    "BusinessTravel": ("Workload", "Reduce travel frequency or rotate travel duty"),
    "MonthlyIncome": ("Pay", "Benchmark pay vs market and consider an adjustment"),
    "StockOptionLevel": ("Pay", "Review equity / long-term incentive eligibility"),
    "PercentSalaryHike": ("Pay", "Review the last raise vs peers in the same role"),
    "JobSatisfaction": ("Engagement", "Manager 1:1 on role fit and career goals"),
    "EnvironmentSatisfaction": ("Engagement", "Run a team/environment pulse check"),
    "JobInvolvement": ("Engagement", "Give ownership of a visible project"),
    "RelationshipSatisfaction": ("Engagement", "Check team dynamics; consider mentor/buddy"),
    "WorkLifeBalance": ("Flexibility", "Offer flexible hours or hybrid days"),
    "DistanceFromHome": ("Flexibility", "Offer remote/hybrid days or commute support"),
    "YearsSinceLastPromotion": ("Growth", "Promotion or role-expansion review"),
    "YearsInCurrentRole": ("Growth", "Role rotation or stretch assignment"),
    "TrainingTimesLastYear": ("Growth", "Enrol in a learning / certification plan"),
    "JobLevel": ("Growth", "Discuss a path to the next level"),
    "YearsWithCurrManager": ("Manager", "Manager-fit check; skip-level conversation or mentor"),
}


def build_pipe(X):
    cat = X.select_dtypes(include="object").columns
    num = X.select_dtypes(exclude="object").columns
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    model = xgb.XGBClassifier(
        objective="binary:logistic", n_estimators=300, max_depth=5,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=5, random_state=42, eval_metric="logloss",
    )
    return Pipeline([("prep", prep), ("model", model)])


def _base_col(feat, cols):
    f = re.sub(r"^(num|cat)__", "", feat)
    for c in sorted(cols, key=len, reverse=True):
        if f.startswith(c):
            return c
    return f


def train(df, exclude_protected=False, k=50):
    """Returns everything the app needs. Risk = out-of-fold probability (honest, not in-sample)."""
    df = df.copy()
    drop = ["Attrition", "EmployeeNumber"] + [c for c in USELESS if c in df.columns]
    if exclude_protected:
        drop += [c for c in PROTECTED if c in df.columns]
    X = df.drop(columns=drop)
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    pipe = build_pipe(X)
    oof = cross_val_predict(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba"
    )[:, 1]
    pipe.fit(X, y)  # final model for explanations / what-if

    # Per-employee driver contributions (TreeSHAP via xgboost), merged back to original columns
    Xt = pipe.named_steps["prep"].transform(X)
    contribs = pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]
    names = pipe.named_steps["prep"].get_feature_names_out()
    base = [_base_col(n, X.columns) for n in names]
    S = pd.DataFrame(contribs, columns=base, index=df.index).T.groupby(level=0).sum().T

    out = df.copy()
    out["RiskScore"] = oof
    top_idx = out["RiskScore"].nlargest(k).index
    metrics = {
        "ROC-AUC": roc_auc_score(y, oof),
        "PR-AUC": average_precision_score(y, oof),
        "Base rate": y.mean(),
        f"Precision@{k}": y.loc[top_idx].mean(),
        f"Recall@{k}": y.loc[top_idx].sum() / y.sum(),
    }
    return dict(df=out, X=X, y=y, pipe=pipe, S=S, metrics=metrics, k=k)


def reasons(art, idx, n=3):
    """Top-n factors pushing risk UP for this employee, in plain words."""
    row, s = art["df"].loc[idx], art["S"].loc[idx]
    top = s[s > 0].sort_values(ascending=False).head(n)
    return [f"{c} = {row[c]}" for c in top.index]


def recommend(art, idx, n=3):
    """Actionable recommendations tied to this employee's own risk drivers."""
    s = art["S"].loc[idx].sort_values(ascending=False)
    recs, seen = [], set()
    for c, v in s.items():
        if v <= 0.05 or c not in ACTIONS:
            continue
        theme, action = ACTIONS[c]
        if theme in seen:
            continue
        seen.add(theme)
        recs.append({"theme": theme, "action": action, "driver": f"{c} = {art['df'].loc[idx, c]}"})
        if len(recs) == n:
            break
    if not recs:
        recs.append({"theme": "General", "action": "Conduct a structured retention conversation", "driver": "-"})
    return recs


def exit_cost(monthly_income):
    """Brief says each exit costs 6-9 months of salary."""
    return 6 * monthly_income, 9 * monthly_income


def watchlist(art, k=50, dept=None):
    d = art["df"]
    if dept and dept != "All":
        d = d[d["Department"] == dept]
    d = d.nlargest(k, "RiskScore").copy()
    d["Top3Reasons"] = [", ".join(reasons(art, i)) for i in d.index]
    d["Recommendation"] = ["; ".join(r["action"] for r in recommend(art, i)) for i in d.index]
    return d


def what_if(art, idx, changes):
    """Re-score one employee with edited fields. Returns (current, new) risk from the final model."""
    row = art["X"].loc[[idx]].copy()
    cur = art["pipe"].predict_proba(row)[0, 1]
    for k, v in changes.items():
        row[k] = v
    return cur, art["pipe"].predict_proba(row)[0, 1]


# ---------- HR Copilot (rule-based, free, deployable anywhere) ----------
HELP = (
    "Try: **employee 622**, **top 10**, **top 5 in Sales**, **who works overtime**, "
    "**low pay risks**, **what should I do for 1494?**"
)


def _employee_md(art, emp):
    d = art["df"]
    row = d[d["EmployeeNumber"] == emp]
    if row.empty:
        return f"I couldn't find employee **{emp}**."
    idx = row.index[0]
    r = row.iloc[0]
    lo, hi = exit_cost(r["MonthlyIncome"])
    lines = [
        f"### Employee {emp} — risk **{r['RiskScore']:.0%}**",
        f"{r['JobRole']} · {r['Department']} · income ${r['MonthlyIncome']:,}/mo · overtime: {r['OverTime']}",
        f"**Why:** " + "; ".join(reasons(art, idx)),
        "**What to do:**",
    ]
    lines += [f"- **{x['theme']}** — {x['action']}  _(driver: {x['driver']})_" for x in recommend(art, idx)]
    lines.append(f"Estimated cost if they leave: ${lo:,.0f}–${hi:,.0f}")
    return "\n".join(lines)


def _table_md(d):
    t = d[["EmployeeNumber", "Department", "JobRole", "RiskScore"]].copy()
    t["RiskScore"] = t["RiskScore"].map("{:.0%}".format)
    return t.to_markdown(index=False)


def copilot(art, q):
    ql = q.lower()
    d = art["df"]
    m_top = re.search(r"top\s*(\d+)", ql)
    m_emp = re.search(r"(?:employee|emp|id|#)\s*#?\s*(\d+)", ql) or (
        None if m_top else re.search(r"\b(\d{1,4})\b", ql)
    )
    if m_emp:
        return _employee_md(art, int(m_emp.group(1)))

    sub = d
    for key, name in [("sales", "Sales"), ("research", "Research & Development"),
                      ("r&d", "Research & Development"), ("hr", "Human Resources"),
                      ("human", "Human Resources")]:
        if re.search(rf"\b{re.escape(key)}\b", ql):
            sub = d[d["Department"] == name]
            break
    if "overtime" in ql:
        sub = sub[sub["OverTime"] == "Yes"]
    if re.search(r"pay|salary|income|compensation", ql):
        sub = sub[sub["MonthlyIncome"] < d["MonthlyIncome"].quantile(0.25)]
    if "promotion" in ql:
        sub = sub[sub["YearsSinceLastPromotion"] >= 4]

    if m_top or sub is not d:
        n = int(m_top.group(1)) if m_top else 10
        sub = sub.nlargest(n, "RiskScore")
        if sub.empty:
            return "No employees match that filter."
        return f"Highest-risk employees matching your request:\n\n{_table_md(sub)}\n\nAsk **employee <id>** for reasons and actions."
    return HELP


# ---------- Cycle planner (enhancement 3) ----------
DEFAULT_COSTS = {
    "raise_pct": 10,            # % of annual salary for a pay raise
    "promo_pct": 8,             # % of annual salary for a promotion (incl. raise)
    "Overtime cut": 1500,       # backfill / extra headcount cost per person
    "Manager career talk": 200,
    "Flexible work": 300,
    "Reduce travel": 500,
    "Training plan": 1000,
}


def _levers(raise_pct):
    travel = {"Travel_Frequently": "Travel_Rarely", "Travel_Rarely": "Non-Travel", "Non-Travel": "Non-Travel"}
    return {
        "Overtime cut": lambda X: X.assign(OverTime="No"),
        "Pay raise": lambda X: X.assign(MonthlyIncome=(X["MonthlyIncome"] * (1 + raise_pct / 100)).round()),
        "Manager career talk": lambda X: X.assign(JobSatisfaction=np.minimum(X["JobSatisfaction"] + 1, 4)),
        "Flexible work": lambda X: X.assign(WorkLifeBalance=np.minimum(X["WorkLifeBalance"] + 1, 4)),
        "Promotion review": lambda X: X.assign(YearsSinceLastPromotion=0),
        "Reduce travel": lambda X: X.assign(BusinessTravel=X["BusinessTravel"].map(travel)),
        "Training plan": lambda X: X.assign(TrainingTimesLastYear=np.minimum(X["TrainingTimesLastYear"] + 2, 6)),
    }


def _cost(lever, income, c):
    if lever == "Pay raise":
        return income * 12 * c["raise_pct"] / 100
    if lever == "Promotion review":
        return income * 12 * c["promo_pct"] / 100
    return c[lever]


def plan(art, k=50, budget=None, costs=None, pool=200):
    """Pick the best single action per employee and fill the cycle by ROI.
    Saved = out-of-fold risk x model-estimated relative risk drop x 7.5 months salary (mid of 6-9)."""
    c = {**DEFAULT_COSTS, **(costs or {})}
    d = art["df"]
    pool_idx = d["RiskScore"].nlargest(pool).index
    Xp = art["X"].loc[pool_idx]
    p0 = art["pipe"].predict_proba(Xp)[:, 1]
    rows = []
    for name, fn in _levers(c["raise_pct"]).items():
        p1 = art["pipe"].predict_proba(fn(Xp.copy()))[:, 1]
        rel = np.clip((p0 - p1) / np.maximum(p0, 1e-6), 0, 1)
        for i, idx in enumerate(pool_idx):
            inc, risk = d.loc[idx, "MonthlyIncome"], d.loc[idx, "RiskScore"]
            rows.append({
                "EmployeeNumber": d.loc[idx, "EmployeeNumber"], "Department": d.loc[idx, "Department"],
                "JobRole": d.loc[idx, "JobRole"], "Action": name, "RiskBefore": risk,
                "RiskAfter": risk * (1 - rel[i]), "Cost": _cost(name, inc, c),
                "Saved": risk * rel[i] * 7.5 * inc,
            })
    cand = pd.DataFrame(rows)
    cand["Net"] = cand["Saved"] - cand["Cost"]
    best = (cand[cand["Net"] > 0].sort_values("Net", ascending=False)
            .drop_duplicates("EmployeeNumber"))
    best["ROI"] = best["Saved"] / best["Cost"]
    best = best.sort_values("ROI", ascending=False)
    chosen, spent = [], 0.0
    for _, r in best.iterrows():
        if len(chosen) >= k:
            break
        if budget is not None and spent + r["Cost"] > budget:
            continue
        chosen.append(r)
        spent += r["Cost"]
    return pd.DataFrame(chosen).reset_index(drop=True)


# ---------- Team view (enhancement 4) ----------
def team_summary(art, by, high=0.5):
    d = art["df"]
    return (d.groupby(by).agg(
        Headcount=("EmployeeNumber", "count"),
        AvgRisk=("RiskScore", "mean"),
        HighRisk=("RiskScore", lambda s: int((s >= high).sum())),
        OvertimePct=("OverTime", lambda s: (s == "Yes").mean()),
        AvgIncome=("MonthlyIncome", "mean"),
        AvgJobSat=("JobSatisfaction", "mean"),
    ).reset_index().sort_values("AvgRisk", ascending=False))


def team_drivers(art, mask, n=3):
    """Most common actionable risk drivers in a team + the matching team-level action."""
    s = art["S"][mask].mean().sort_values(ascending=False)
    out = []
    for c, v in s.items():
        if v > 0.02 and c in ACTIONS:
            out.append({"driver": c, "avg_push": round(float(v), 3), "action": ACTIONS[c][1]})
        if len(out) == n:
            break
    return out


# ---------- LLM context (enhancement 2) ----------
LLM_SYSTEM = (
    "You are TalentGuard's HR retention advisor for TechNova Solutions. Answer ONLY from the data below; "
    "never invent employees, numbers or policies. Risk scores are statistical estimates, not facts about a person. "
    "Give concrete, humane retention actions (workload, pay, growth, flexibility, manager) and a short reason. "
    "Never recommend adverse action based on the score, and do not base advice on age, gender or marital status. "
    "If the data does not contain the answer, say so. Be concise (under 200 words unless asked for more).\n\nDATA:\n"
)


def llm_context(art, question, n=25):
    d = art["df"]
    wl = watchlist(art, n)
    lines = [f"{len(d)} employees; actual attrition {art['metrics']['Base rate']:.1%}.",
             "Average risk by department: " + ", ".join(
                 f"{k} {v:.0%}" for k, v in d.groupby("Department")["RiskScore"].mean().items()),
             f"Top {n} highest-risk employees (id | dept | role | risk | reasons | suggested actions):"]
    in_wl = set(wl["EmployeeNumber"])
    for _, r in wl.iterrows():
        lines.append(f"{r['EmployeeNumber']} | {r['Department']} | {r['JobRole']} | {r['RiskScore']:.0%} | "
                     f"{r['Top3Reasons']} | {r['Recommendation']}")
    for m in set(re.findall(r"\b\d{1,4}\b", question)):
        e = int(m)
        if e not in in_wl and (d["EmployeeNumber"] == e).any():
            lines.append("\nRequested employee:\n" + _employee_md(art, e))
    return LLM_SYSTEM + "\n".join(lines)
