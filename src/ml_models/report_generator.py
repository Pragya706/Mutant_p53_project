"""
Clinical-style PDF report generator for p53 mutation analysis.

Usage:
    from report_generator import generate_report
    pdf_bytes = generate_report(patient_data, prediction_result)
"""

import csv
import io
import os
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#12355B")
BLUE = colors.HexColor("#2563A6")
LIGHT_BLUE = colors.HexColor("#EAF3FF")
PALE_BLUE = colors.HexColor("#F5F9FF")
GREEN = colors.HexColor("#0F8A5F")
RED = colors.HexColor("#B42318")
AMBER = colors.HexColor("#B7791F")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#5B6675")
BORDER = colors.HexColor("#D6DEE8")
SOFT = colors.HexColor("#F7F9FC")
WHITE = colors.white


def _safe(value, default="Not provided"):
    text = str(value or "").strip()
    return escape(text) if text else default


def _pct(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def _style_sheet():
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#DCEBFF"),
            alignment=TA_LEFT,
        ),
        "right_header": ParagraphStyle(
            "right_header",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=WHITE,
            alignment=TA_RIGHT,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=INK,
        ),
        "body_small": ParagraphStyle(
            "body_small",
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "value": ParagraphStyle(
            "value",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=INK,
        ),
        "kpi": ParagraphStyle(
            "kpi",
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label",
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica",
            fontSize=7.2,
            leading=9.5,
            textColor=MUTED,
        ),
    }


def _cell(label, value, styles):
    return [Paragraph(label, styles["label"]), Paragraph(value, styles["value"])]


def _table_style(header=True):
    base = [
        ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]
    if header:
        base.extend([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
            ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    return TableStyle(base)


def _load_model_rows():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "results", "model_comparison.csv")
    if not os.path.exists(path):
        return [
            ["GNN", "N/A", "N/A", "N/A", "N/A"],
            ["XGBoost", "N/A", "N/A", "N/A", "N/A"],
            ["Random Forest", "N/A", "N/A", "N/A", "N/A"],
            ["Ensemble", "N/A", "N/A", "N/A", "N/A"],
        ]

    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append([
                    row.get("model", "N/A"),
                    _fmt_metric(row.get("auc")),
                    _fmt_metric(row.get("f1")),
                    _fmt_metric(row.get("precision")),
                    _fmt_metric(row.get("accuracy")),
                ])
    except Exception:
        return [["Metrics file", "Could not read", "", "", ""]]
    return rows or [["Metrics file", "No rows found", "", "", ""]]


def _fmt_metric(value):
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "N/A"


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.line(doc.leftMargin, 12 * mm, A4[0] - doc.rightMargin, 12 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        doc.leftMargin,
        8 * mm,
        "AI research prototype - not for clinical decision-making",
    )
    canvas.drawRightString(
        A4[0] - doc.rightMargin,
        8 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def generate_report(patient: dict, result: dict) -> bytes:
    patient = patient or {}
    result = result or {}

    buf = io.BytesIO()
    margin = 15 * mm
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=13 * mm,
        bottomMargin=16 * mm,
        title="p53 Mutation Analysis Report",
        author="Group 64 MSRUAS",
    )

    styles = _style_sheet()
    story = []
    usable_w = A4[0] - 2 * margin
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    report_id = datetime.now().strftime("P53-%Y%m%d-%H%M%S")
    mutation = (
        f"{patient.get('WT_AA_1', '?')}"
        f"{patient.get('Codon', '?')}"
        f"{patient.get('Mutant_AA_1', '?')}"
    )
    pathogenicity = str(result.get("pathogenicity", "Unknown"))
    is_pathogenic = pathogenicity.lower() == "pathogenic"
    result_color = RED if is_pathogenic else GREEN
    patient_id = _safe(patient.get("Patient_ID"))
    notes = _safe(patient.get("Notes"))
    hotspot = "Yes" if patient.get("Codon") in {175, 245, 248, 249, 273, 282} else "No"

    header = Table(
        [[
            [
                Paragraph("p53 Mutation Analysis Report", styles["title"]),
                Paragraph("Graph-Based Federated AI for Mutant p53 Reactivation", styles["subtitle"]),
            ],
            Paragraph(f"Report ID<br/>{report_id}<br/><br/>Generated<br/>{now}", styles["right_header"]),
        ]],
        colWidths=[usable_w * 0.68, usable_w * 0.32],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(header)
    story.append(Spacer(1, 8))

    patient_band = Table(
        [[
            _cell("PATIENT / SAMPLE", patient_id, styles),
            _cell("MUTATION", escape(mutation), styles),
            _cell("CANCER / DISEASE", _safe(patient.get("Disease")), styles),
            _cell("SAMPLE ORIGIN", _safe(patient.get("Mutation_origin")), styles),
        ]],
        colWidths=[usable_w * 0.30, usable_w * 0.18, usable_w * 0.27, usable_w * 0.25],
    )
    patient_band.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_band)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Result Summary", styles["section"]))
    kpi = Table(
        [[
            [Paragraph(pathogenicity, ParagraphStyle("path_kpi", parent=styles["kpi"], textColor=result_color)),
             Paragraph("Predicted classification", styles["kpi_label"])],
            [Paragraph(_pct(result.get("confidence")), styles["kpi"]),
             Paragraph("Prediction confidence", styles["kpi_label"])],
            [Paragraph(_pct(result.get("restoration_score")), styles["kpi"]),
             Paragraph("Restoration priority", styles["kpi_label"])],
            [Paragraph(hotspot, styles["kpi"]),
             Paragraph("Known hotspot codon", styles["kpi_label"])],
        ]],
        colWidths=[usable_w / 4] * 4,
    )
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WHITE),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("LINEABOVE", (0, 0), (0, 0), 3, result_color),
        ("LINEABOVE", (1, 0), (1, 0), 3, BLUE),
        ("LINEABOVE", (2, 0), (2, 0), 3, AMBER),
        ("LINEABOVE", (3, 0), (3, 0), 3, NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(kpi)
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "Interpretation: this report summarizes model-generated research evidence "
        "for TP53 mutation pathogenicity and possible protein restoration targets. "
        "It is formatted like a clinical report for readability, but it remains an "
        "academic AI output and must not be used as a medical diagnosis.",
        styles["body"],
    ))

    story.append(Paragraph("Patient Notes", styles["section"]))
    notes_box = Table(
        [[Paragraph(notes.replace("\n", "<br/>"), styles["body"])]],
        colWidths=[usable_w],
    )
    notes_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.7, BORDER),
        ("LINEBEFORE", (0, 0), (-1, -1), 3, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(notes_box)

    story.append(Paragraph("Mutation Details", styles["section"]))
    details = [
        ["Field", "Value", "Field", "Value"],
        ["Codon position", _safe(patient.get("Codon")), "Mutation type", _safe(patient.get("Type"))],
        ["Wild-type amino acid", _safe(patient.get("WT_AA_1")), "Mutant amino acid", _safe(patient.get("Mutant_AA_1"))],
        ["Protein domain", _safe(patient.get("Domain")), "CpG site", _safe(patient.get("CpG"))],
        ["Mutation label", escape(mutation), "Hotspot", hotspot],
    ]
    details_tbl = Table(details, colWidths=[usable_w * 0.25] * 4)
    details_tbl.setStyle(_table_style())
    story.append(details_tbl)

    story.append(Paragraph("Protein Target Ranking", styles["section"]))
    target_rows = [["Rank", "Protein", "Score", "Suggested mechanism"]]
    for idx, target in enumerate(result.get("top_targets", [])[:5], 1):
        target_rows.append([
            str(idx),
            _safe(target.get("protein")),
            _fmt_metric(target.get("score")),
            _safe(target.get("mechanism")),
        ])
    if len(target_rows) == 1:
        target_rows.append(["-", "No target data", "-", "-"])
    target_tbl = Table(
        target_rows,
        colWidths=[usable_w * 0.08, usable_w * 0.17, usable_w * 0.13, usable_w * 0.62],
    )
    target_tbl.setStyle(_table_style())
    story.append(target_tbl)

    story.append(Paragraph("Model Evidence", styles["section"]))
    model_rows = [["Model", "AUC", "F1", "Precision", "Accuracy"]] + _load_model_rows()
    model_tbl = Table(
        model_rows,
        colWidths=[usable_w * 0.28, usable_w * 0.16, usable_w * 0.16, usable_w * 0.20, usable_w * 0.20],
    )
    model_tbl.setStyle(_table_style())
    story.append(model_tbl)

    story.append(Paragraph("Methodology", styles["section"]))
    story.append(Paragraph(
        "The pipeline combines mutation-level TP53 features, protein-protein "
        "interaction information from STRING/BioGRID, graph neural network outputs, "
        "classical machine-learning models, and federated-learning simulation. "
        "Protein rankings are produced from model confidence, hotspot status, "
        "domain context, and target-interaction signals.",
        styles["body"],
    ))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width=usable_w, thickness=0.8, color=BORDER))
    story.append(Paragraph(
        "Disclaimer: This document is generated by an AI research prototype for "
        "academic project demonstration. It is not a diagnostic laboratory report, "
        "not a prescription, and not clinical medical advice. Any medical concern "
        "must be reviewed by qualified healthcare professionals.",
        styles["disclaimer"],
    ))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    return buf.read()


if __name__ == "__main__":
    sample_patient = {
        "Patient_ID": "Patient-001",
        "Notes": "Example clinical note entered by the user.",
        "Codon": 175,
        "Type": "Missense",
        "CpG": "Yes",
        "Domain": "L2",
        "WT_AA_1": "R",
        "Mutant_AA_1": "H",
        "Mutation_origin": "Somatic",
        "Disease": "Breast",
    }
    sample_result = {
        "pathogenicity": "Pathogenic",
        "confidence": 0.872,
        "restoration_score": 0.828,
        "top_targets": [
            {"protein": "HSP90", "score": 0.818, "mechanism": "Chaperone pathway target"},
            {"protein": "MDM2", "score": 0.668, "mechanism": "Ubiquitin ligase inhibition pathway"},
        ],
    }
    with open("test_report.pdf", "wb") as f:
        f.write(generate_report(sample_patient, sample_result))
