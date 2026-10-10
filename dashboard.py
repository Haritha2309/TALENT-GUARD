from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core
import llm
import reports
import tracker

st.set_page_config(page_title="TalentGuard", page_icon="🛡️", layout="wide")
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
LOG_PATH = ROOT / "data" / "action_log.csv"


def secret(name, default=None):
    try:
        return st.secrets[name] if name in st.secrets else default
    except Exception:  # no secrets.toml
        return default


@st.cache_resource(show_spinner="Training model (one-time, ~20s)...")
def get_art(key, exclude_protected, _df):
    return core.train(_df, exclude_protected=exclude_protected)


@st.cache_data(show_spinner="Running planner...")
def run_plan(key, excl, _art, k, budget, costs):
    return core.plan(_art, k=k, budget=budget or None, costs=dict(costs))


# ---------------- sidebar ----------------
st.sidebar.title("🛡️ TalentGuard")
up = st.sidebar.file_uploader("Use a different HR CSV (same columns)", type="csv")
raw = pd.read_csv(up) if up else pd.read_csv(DATA)
excl = st.sidebar.checkbox("Exclude Age / Gender / Marital status from model", value=False,
                           help="Fairness option: removes protected attributes before training.")
K = st.sidebar.slider("Retention talks per cycle", 10, 100, 50, 5)
dept = st.sidebar.selectbox("Department", ["All"] + sorted(raw["Department"].unique()))
owner = st.sidebar.text_input("Your name (for the action log)", "HR")

KEY = int(pd.util.hash_pandas_object(raw).sum() % 10**9)
art = get_art(KEY, excl, raw)
df = art["df"]
df["Team"] = df["Department"] + " · " + df["JobRole"]
df["MgrTenure"] = pd.cut(df["YearsWithCurrManager"], [-1, 0, 2, 5, 100],
                         labels=["New (0 yrs)", "1-2 yrs", "3-5 yrs", "6+ yrs"]).astype(str)

API_KEY = llm.get_key(st.secrets)
st.sidebar.caption("🤖 Claude copilot: " + ("connected" if API_KEY else "no API key (rule-based mode)"))

if "log" not in st.session_state:
    st.session_state.log = tracker.load(LOG_PATH)


def log_items(items):
    st.session_state.log, n = tracker.add(st.session_state.log, items, owner)
    tracker.save(st.session_state.log, LOG_PATH)
    return n


def brief_pdf(idx):
    r = df.loc[idx]
    return reports.employee_pdf(r, r["RiskScore"], core.reasons(art, idx), core.recommend(art, idx),
                                core.exit_cost(r["MonthlyIncome"]), art["S"].loc[idx])


st.title("Attrition Risk Dashboard")
tabs = st.tabs(["Overview", "Watch list", "Employee deep-dive", "Cycle planner",
                "Teams", "Action tracker", "Ask HR Copilot"])

# ---------------- overview ----------------
with tabs[0]:
    m = art["metrics"]
    c = st.columns(5)
    c[0].metric("Employees", f"{len(df):,}")
    c[1].metric("Actual attrition", f"{m['Base rate']:.1%}")
    c[2].metric("ROC-AUC (5-fold)", f"{m['ROC-AUC']:.2f}")
    c[3].metric("PR-AUC (5-fold)", f"{m['PR-AUC']:.2f}", help="The metric that matters at ~1 in 6 attrition")
    c[4].metric("Precision@50", f"{m['Precision@50']:.0%}", help="Of the 50 flagged, share who truly left")
    st.caption("All scores are out-of-fold: each employee is scored by a model that never saw them.")
    left, right = st.columns(2)
    left.subheader("Average risk by department")
    left.bar_chart(df.groupby("Department")["RiskScore"].mean())
    right.subheader("Average risk by job role")
    right.bar_chart(df.groupby("JobRole")["RiskScore"].mean().sort_values())
    top = df.nlargest(K, "RiskScore")
    st.info(f"Exposure if the top {K} all resigned: **${6 * top['MonthlyIncome'].sum():,.0f} – "
            f"${9 * top['MonthlyIncome'].sum():,.0f}** (6–9 months of salary each).")

# ---------------- watch list ----------------
with tabs[1]:
    wl = core.watchlist(art, K, dept)
    show = wl[["EmployeeNumber", "Department", "JobRole", "RiskScore", "Top3Reasons"]].copy()
    show["Recommendation"] = wl["Recommendation"].where(wl["RiskScore"].rank(ascending=False) <= 10, "")
    st.subheader(f"Top {len(wl)} watch list" + ("" if dept == "All" else f" — {dept}"))
    st.dataframe(show.style.format({"RiskScore": "{:.1%}"}).background_gradient(subset=["RiskScore"], cmap="Reds"),
                 use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ top50_watchlist.csv",
                       wl[["EmployeeNumber", "RiskScore", "Top3Reasons"]].to_csv(index=False), "top50_watchlist.csv")
    c2.download_button("⬇ top10_recommendations.csv",
                       wl.head(10)[["EmployeeNumber", "RiskScore", "Top3Reasons", "Recommendation"]].to_csv(index=False),
                       "top10_recommendations.csv")
    if c3.button("📄 Prepare PDF briefs for top 10"):
        with st.spinner("Building PDFs..."):
            st.session_state.zip = reports.zip_pdfs(
                {f"brief_{df.loc[i, 'EmployeeNumber']}.pdf": brief_pdf(i) for i in wl.head(10).index})
    if "zip" in st.session_state:
        c3.download_button("⬇ top10_briefs.zip", st.session_state.zip, "top10_briefs.zip")

# ---------------- employee deep-dive ----------------
with tabs[2]:
    ids = df.sort_values("RiskScore", ascending=False)["EmployeeNumber"].tolist()
    emp = st.selectbox("Employee", ids, key="emp_sel")
    idx = df.index[df["EmployeeNumber"] == emp][0]
    r = df.loc[idx]
    recs = core.recommend(art, idx)
    a, b = st.columns(2)
    with a:
        st.metric("Risk score", f"{r['RiskScore']:.0%}")
        st.write(f"**{r['JobRole']}** · {r['Department']} · ${r['MonthlyIncome']:,}/mo · {r['YearsAtCompany']} yrs at company")
        st.markdown("**Recommended actions**")
        for x in recs:
            st.markdown(f"- **{x['theme']}** — {x['action']}  \n  <small>driver: {x['driver']}</small>",
                        unsafe_allow_html=True)
        lo, hi = core.exit_cost(r["MonthlyIncome"])
        st.caption(f"Cost if they leave: ${lo:,.0f} – ${hi:,.0f}")
        b1, b2 = st.columns(2)
        b1.download_button("📄 PDF brief", brief_pdf(idx), f"brief_{emp}.pdf", mime="application/pdf")
        if b2.button("➕ Log these actions"):
            n = log_items([(emp, x["action"], r["RiskScore"]) for x in recs])
            st.success(f"{n} action(s) added to the tracker.")
    with b:
        s = art["S"].loc[idx]
        s = s.reindex(s.abs().sort_values(ascending=False).index)[:8][::-1]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh([f"{c} = {r[c]}" for c in s.index], s.values,
                color=["#d62728" if v > 0 else "#2ca02c" for v in s.values])
        ax.set_title("Red raises risk, green lowers it")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("#### What-if simulator")
    st.caption("Change levers and see how the model's risk moves (correlation, not a guarantee).")
    w1, w2, w3 = st.columns(3)
    trav = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    ot = w1.selectbox("OverTime", ["Yes", "No"], index=0 if r["OverTime"] == "Yes" else 1)
    inc = w1.number_input("MonthlyIncome", 1000, 30000, int(r["MonthlyIncome"]), 500)
    js = w2.slider("JobSatisfaction", 1, 4, int(r["JobSatisfaction"]))
    wlb = w2.slider("WorkLifeBalance", 1, 4, int(r["WorkLifeBalance"]))
    bt = w3.selectbox("BusinessTravel", trav, index=trav.index(r["BusinessTravel"]))
    ysp = w3.slider("YearsSinceLastPromotion", 0, 15, int(r["YearsSinceLastPromotion"]))
    cur, new = core.what_if(art, idx, dict(OverTime=ot, MonthlyIncome=inc, JobSatisfaction=js,
                                           WorkLifeBalance=wlb, BusinessTravel=bt, YearsSinceLastPromotion=ysp))
    st.metric("Simulated risk", f"{new:.0%}", f"{new - cur:+.0%} vs current model score", delta_color="inverse")

# ---------------- cycle planner ----------------
with tabs[3]:
    st.subheader("Cycle planner — best use of this cycle's talks and budget")
    st.caption("For each high-risk employee the model tests every lever (overtime cut, raise, promotion, ...), "
               "keeps the one with the best payoff, then fills the cycle by return on spend.")
    p1, p2 = st.columns(2)
    k_plan = p1.slider("Talks / actions this cycle", 5, 100, K, 5, key="kplan")
    budget = p2.number_input("Budget ($, 0 = no limit)", 0, 5_000_000, 100_000, 10_000)
    with st.expander("Cost assumptions (edit to match your company)"):
        cc = st.columns(4)
        costs = {
            "raise_pct": cc[0].number_input("Raise % of salary", 1, 30, core.DEFAULT_COSTS["raise_pct"]),
            "promo_pct": cc[1].number_input("Promotion % of salary", 1, 30, core.DEFAULT_COSTS["promo_pct"]),
            "Overtime cut": cc[2].number_input("Overtime cut $", 0, 20000, core.DEFAULT_COSTS["Overtime cut"], 100),
            "Manager career talk": cc[3].number_input("Career talk $", 0, 5000, core.DEFAULT_COSTS["Manager career talk"], 50),
            "Flexible work": cc[0].number_input("Flexible work $", 0, 5000, core.DEFAULT_COSTS["Flexible work"], 50),
            "Reduce travel": cc[1].number_input("Reduce travel $", 0, 5000, core.DEFAULT_COSTS["Reduce travel"], 50),
            "Training plan": cc[2].number_input("Training plan $", 0, 10000, core.DEFAULT_COSTS["Training plan"], 100),
        }
    pl = run_plan(KEY, excl, art, k_plan, int(budget), tuple(sorted(costs.items())))
    if pl.empty:
        st.warning("No action pays for itself under these assumptions. Try a larger budget or lower costs.")
    else:
        k1 = st.columns(4)
        k1[0].metric("Employees covered", len(pl))
        k1[1].metric("Total spend", f"${pl['Cost'].sum():,.0f}")
        k1[2].metric("Expected exits avoided (value)", f"${pl['Saved'].sum():,.0f}")
        k1[3].metric("Net benefit", f"${pl['Net'].sum():,.0f}")
        view = pl[["EmployeeNumber", "Department", "JobRole", "Action", "RiskBefore", "RiskAfter",
                   "Cost", "Saved", "ROI"]]
        st.dataframe(view.style.format({"RiskBefore": "{:.0%}", "RiskAfter": "{:.0%}", "Cost": "${:,.0f}",
                                        "Saved": "${:,.0f}", "ROI": "{:.1f}x"}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(pl.groupby("Action")["Cost"].sum())
        d1, d2 = st.columns(2)
        d1.download_button("⬇ cycle_plan.csv", view.to_csv(index=False), "cycle_plan.csv")
        if d2.button("➕ Add this plan to the tracker"):
            n = log_items([(r.EmployeeNumber, r.Action, r.RiskBefore) for r in pl.itertuples()])
            st.success(f"{n} action(s) added to the tracker.")
    st.caption("Estimates assume the model's lever effects are causal and exit cost = 7.5 months of salary. "
               "Treat as a prioritisation aid, then validate with the retention tracker.")

# ---------------- teams ----------------
with tabs[4]:
    st.subheader("Team & manager hot spots")
    st.caption("The IBM dataset has no manager ID, so teams are Department / Job role, and managers are proxied by "
               "how long the employee has had their current manager. If your own CSV has a `ManagerID` column, "
               "it appears below automatically.")
    options = {"Department": "Department", "Job role": "JobRole", "Department · Role": "Team",
               "Years with current manager": "MgrTenure"}
    if "ManagerID" in df.columns:
        options = {"Manager": "ManagerID", **options}
    t1, t2 = st.columns(2)
    label = t1.selectbox("Group by", list(options))
    hi_thr = t2.slider("'High risk' threshold", 0.2, 0.9, 0.5, 0.05)
    col = options[label]
    ts = core.team_summary(art, col, hi_thr)
    st.dataframe(ts.style.format({"AvgRisk": "{:.1%}", "OvertimePct": "{:.0%}", "AvgIncome": "${:,.0f}",
                                  "AvgJobSat": "{:.2f}"}).background_gradient(subset=["AvgRisk"], cmap="Reds"),
                 use_container_width=True, hide_index=True)
    st.markdown("**Heat map: average risk, job role × department**")
    st.dataframe(df.pivot_table(index="JobRole", columns="Department", values="RiskScore", aggfunc="mean")
                 .style.format("{:.0%}", na_rep="-").background_gradient(cmap="Reds", axis=None),
                 use_container_width=True)
    pick = st.selectbox("Drill into a team", ts[col].tolist())
    mask = df[col] == pick
    st.write(f"**{pick}** — {int(mask.sum())} people, average risk {df.loc[mask, 'RiskScore'].mean():.0%}")
    drv = core.team_drivers(art, mask)
    if drv:
        st.markdown("**Team-level actions** (what pushes risk up most across this team)")
        for x in drv:
            st.markdown(f"- {x['action']}  \n  <small>driver: {x['driver']}</small>", unsafe_allow_html=True)
    else:
        st.write("No dominant actionable driver for this team.")
    st.dataframe(df[mask].nlargest(10, "RiskScore")[["EmployeeNumber", "JobRole", "RiskScore", "OverTime", "MonthlyIncome"]]
                 .style.format({"RiskScore": "{:.0%}"}), use_container_width=True, hide_index=True)

# ---------------- action tracker ----------------
with tabs[5]:
    st.subheader("Retention action tracker")
    sm = tracker.summary(st.session_state.log)
    t = st.columns(4)
    t[0].metric("Actions logged", sm["actions"])
    t[1].metric("Completed", f"{sm['done_pct']:.0%}")
    t[2].metric("Outcomes known", sm["known"])
    if sm["stay_rate"] is not None:
        t[3].metric("Stayed (acted on)", f"{sm['stay_rate']:.0%}",
                    f"{sm['stay_rate'] - sm['expected_stay']:+.0%} vs model expectation")
    else:
        t[3].metric("Stayed (acted on)", "-")
    with st.expander("➕ Log an action manually"):
        f1, f2 = st.columns(2)
        e = f1.selectbox("Employee", ids, key="log_emp")
        act = f2.text_input("Action", "Manager career discussion")
        if st.button("Add to log"):
            risk = float(df.loc[df["EmployeeNumber"] == e, "RiskScore"].iloc[0])
            log_items([(e, act, risk)])
            st.rerun()
    edited = st.data_editor(
        st.session_state.log, num_rows="dynamic", use_container_width=True, hide_index=True, key="log_editor",
        column_config={"Status": st.column_config.SelectboxColumn(options=tracker.STATUS),
                       "Outcome": st.column_config.SelectboxColumn(options=tracker.OUTCOME)})
    if not edited.reset_index(drop=True).equals(st.session_state.log.reset_index(drop=True)):
        st.session_state.log = edited.reset_index(drop=True)
        tracker.save(st.session_state.log, LOG_PATH)
    g1, g2 = st.columns(2)
    g1.download_button("⬇ Backup log (CSV)", st.session_state.log.to_csv(index=False), "action_log.csv")
    restore = g2.file_uploader("Restore from backup", type="csv", label_visibility="collapsed")
    if restore is not None and st.button("Load uploaded log"):
        st.session_state.log = pd.read_csv(restore)[tracker.COLS]
        tracker.save(st.session_state.log, LOG_PATH)
        st.rerun()
    st.caption("On Streamlit Community Cloud the log resets when the app restarts: download a backup after each "
               "session (or connect a Google Sheet / database for permanent storage). 'Stayed vs model expectation' "
               "is a rough signal, not proof that the actions caused the result.")

# ---------------- copilot ----------------
with tabs[6]:
    st.subheader("Ask HR Copilot")
    mode = "Rules"
    if API_KEY:
        mode = st.radio("Mode", ["AI (Claude)", "Rules"], horizontal=True)
        if mode.startswith("AI"):
            st.caption("Employee IDs, roles, risk scores and reasons are sent to the Claude API to answer. "
                       "No names are in this dataset.")
    else:
        st.caption(core.HELP + "  — add `ANTHROPIC_API_KEY` to Streamlit secrets for free-text AI answers.")
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! Ask me who is at risk, why, and what HR should do.")]
    for role, msg in st.session_state.chat:
        st.chat_message(role).markdown(msg)
    if q := st.chat_input("e.g. Which Sales employees need attention first, and what should I offer?"):
        st.session_state.chat.append(("user", q))
        reply = None
        if API_KEY and mode.startswith("AI"):
            try:
                reply = llm.ask(API_KEY, core.llm_context(art, q), st.session_state.chat[1:],
                                secret("ANTHROPIC_MODEL", llm.DEFAULT_MODEL))
            except Exception as ex:
                st.warning(f"AI mode failed ({type(ex).__name__}); using rule-based answer.")
        st.session_state.chat.append(("assistant", reply or core.copilot(art, q)))
        st.rerun()
