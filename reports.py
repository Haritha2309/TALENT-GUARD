"""One-page PDF retention brief per employee (reportlab + matplotlib only)."""
import io
import zipfile
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def driver_chart_png(row, shap_series, n=8):
    s = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)[:n][::-1]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh([f"{c} = {row[c]}" for c in s.index], s.values,
            color=["#d62728" if v > 0 else "#2ca02c" for v in s.values])
    ax.set_title("What drives this employee's risk (red = raises, green = lowers)", fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def employee_pdf(row, risk, reasons, recs, cost_range, shap_series) -> bytes:
    """row: employee record; recs: list of dicts(theme, action, driver)."""
    st = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.5 * cm)
    level = "HIGH" if risk >= 0.7 else "MEDIUM" if risk >= 0.4 else "LOW"
    el = [Paragraph(f"TalentGuard - Retention Brief: Employee {row['EmployeeNumber']}", st["Title"]),
          Paragraph(f"Generated {date.today():%d %b %Y} - CONFIDENTIAL, for HR and line manager only", st["Italic"]),
          Spacer(1, 10)]

    prof = [["Risk score", f"{risk:.0%}  ({level})"],
            ["Role", f"{row['JobRole']} - {row['Department']}"],
            ["Monthly income", f"${row['MonthlyIncome']:,}"],
            ["Tenure", f"{row['YearsAtCompany']} yrs company / {row['YearsInCurrentRole']} yrs in role"],
            ["Overtime / travel", f"{row['OverTime']} / {row['BusinessTravel']}"],
            ["Cost if they leave", f"${cost_range[0]:,.0f} - ${cost_range[1]:,.0f}"]]
    t = Table(prof, colWidths=[4.5 * cm, 12 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                           ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#c0392b" if level == "HIGH" else "#333333")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    el += [t, Spacer(1, 12), Paragraph("Top risk factors", st["Heading2"]),
           Paragraph("; ".join(reasons) if reasons else "No single dominant factor.", st["BodyText"]),
           Spacer(1, 6), Image(driver_chart_png(row, shap_series), width=14 * cm, height=7 * cm),
           Paragraph("Recommended actions", st["Heading2"])]
    for r in recs:
        el.append(Paragraph(f"<b>{r['theme']}</b> - {r['action']} <font size=8 color='grey'>(driver: {r['driver']})</font>",
                            st["BodyText"], bulletText="\u2022"))
    el += [Spacer(1, 8), Paragraph("Conversation guide", st["Heading2"]),
           Paragraph("Open with appreciation and ask what would make the role better. Do not mention the risk score "
                     "or the model. Agree one concrete action and a follow-up date.", st["BodyText"]),
           Spacer(1, 10),
           Paragraph("The score is a statistical estimate from past patterns, not a prediction about the individual. "
                     "Use it to prioritise conversations, never for adverse decisions.", st["Italic"])]
    doc.build(el)
    return buf.getvalue()


def zip_pdfs(files: dict) -> bytes:
    """files: {filename: pdf_bytes}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()
