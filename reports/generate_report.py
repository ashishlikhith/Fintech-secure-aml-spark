"""Generate the INT315 report from the executed AML Spark notebook."""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as PdfImage,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NOTEBOOK_PATH = ROOT / "notebooks" / "01_aml_spark_workflow.ipynb"
DOCX_PATH = REPORTS / "INT315_Cluster_Computing_Project_Report.docx"
PDF_PATH = REPORTS / "INT315_Cluster_Computing_Project_Report.pdf"

METRIC_COLUMNS = {
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "tp",
    "tn",
    "fp",
    "fn",
    "threshold",
}


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.loc[:, [not column.casefold().startswith("unnamed:") for column in frame.columns]]


def output_text(output: dict[str, Any]) -> str:
    if output.get("output_type") == "stream":
        return "".join(output.get("text", []))
    data = output.get("data", {})
    if "text/plain" in data:
        return "".join(data["text/plain"])
    return ""


def output_tables(output: dict[str, Any]) -> list[pd.DataFrame]:
    html = "".join(output.get("data", {}).get("text/html", []))
    if not html:
        return []
    try:
        return [clean_frame(frame) for frame in pd.read_html(io.StringIO(html))]
    except ValueError:
        return []


def table_for(
    tables: list[pd.DataFrame], required_columns: set[str]
) -> pd.DataFrame | None:
    for frame in tables:
        columns = {str(column).strip().casefold().replace(" ", "_") for column in frame.columns}
        if required_columns.issubset(columns):
            frame = frame.copy()
            frame.columns = [
                str(column).strip().casefold().replace(" ", "_")
                for column in frame.columns
            ]
            return frame
    return None


def load_notebook_evidence(image_directory: Path) -> dict[str, Any]:
    if not NOTEBOOK_PATH.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_PATH}")
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    code_cells = [
        (index, cell)
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "".join(cell.get("source", [])).strip()
    ]
    if not code_cells:
        raise ValueError(f"No Python code cells found in {NOTEBOOK_PATH}")

    cell_tables: dict[int, list[pd.DataFrame]] = {}
    cell_text: dict[int, str] = {}
    images: list[dict[str, Any]] = []
    all_tables: list[pd.DataFrame] = []
    for index, cell in enumerate(cells):
        tables: list[pd.DataFrame] = []
        text_parts: list[str] = []
        for output_index, output in enumerate(cell.get("outputs", [])):
            text = output_text(output).strip()
            if text:
                text_parts.append(text)
            for frame in output_tables(output):
                tables.append(frame)
                all_tables.append(frame)
            image_data = output.get("data", {}).get("image/png")
            if image_data:
                if isinstance(image_data, list):
                    image_data = "".join(image_data)
                image_path = image_directory / f"notebook_cell_{index}_output_{output_index}.png"
                image_path.write_bytes(base64.b64decode(image_data))
                images.append(
                    {
                        "cell": index,
                        "path": image_path,
                        "caption": f"Inline chart output from notebook cell {index + 1}.",
                    }
                )
        cell_tables[index] = tables
        cell_text[index] = "\n".join(text_parts)

    metrics = table_for(all_tables, METRIC_COLUMNS)
    if metrics is None:
        raise ValueError("The notebook does not contain its model metrics output table.")
    for column in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"):
        metrics[column] = pd.to_numeric(metrics[column], errors="raise")
    for column in ("tp", "tn", "fp", "fn"):
        metrics[column] = pd.to_numeric(metrics[column], errors="raise").astype(int)

    class_distribution = table_for(
        all_tables, {"class_name", "count", "percentage"}
    )
    null_labels = [
        "prepared_rows",
        "transaction_id_nulls",
        "timestamp_nulls",
        "sender_account_nulls",
        "receiver_account_nulls",
        "amount_nulls",
        "payment_type_nulls",
        "is_laundering_nulls",
        "is_near_threshold_nulls",
    ]
    null_text = cell_text.get(10, "")
    null_values = {
        label: int(match.group(1))
        for label in null_labels
        if (match := re.search(rf"\b{label}\s+(\d+)\b", null_text))
    }
    null_summary = (
        pd.DataFrame(
            [{"check": label, "count": null_values[label]} for label in null_labels]
        )
        if len(null_values) == len(null_labels)
        else None
    )
    feature_summary = table_for(
        all_tables,
        {
            "is_near_threshold",
            "transaction_count",
            "minimum_amount",
            "average_amount",
            "maximum_amount",
        },
    )
    split_summary = table_for(
        all_tables, {"split", "rows", "positive", "negative", "positive_rate"}
    )
    if class_distribution is None or null_summary is None or split_summary is None:
        raise ValueError("Notebook outputs are missing a required sample, null, or split table.")

    sql_table_sets = {
        "active_senders": {"sender_account", "transaction_count", "total_sent"},
        "aml_rate_by_payment_type": {
            "payment_type",
            "transaction_count",
            "aml_positive",
            "aml_rate",
        },
        "high_value_transactions": {
            "transaction_id",
            "sender_account",
            "receiver_account",
            "amount",
            "payment_type",
        },
        "payment_types": {"payment_type"},
        "sender_account_join": {
            "transaction_id",
            "sender_account",
            "amount",
            "transaction_count",
            "average_amount",
        },
        "transaction_statistics": {
            "transaction_count",
            "total_amount",
            "average_amount",
            "minimum_amount",
            "maximum_amount",
        },
        "wildcard_payment_search": {"transaction_id", "payment_type", "amount"},
    }
    sql_frames = cell_tables.get(18, [])
    sql: dict[str, pd.DataFrame] = {}
    for query_name, columns in sql_table_sets.items():
        match = table_for(sql_frames, columns)
        if match is not None:
            sql[query_name] = match
    if set(sql) != set(sql_table_sets):
        missing = sorted(set(sql_table_sets) - set(sql))
        raise ValueError(f"Notebook SQL outputs are missing tables: {', '.join(missing)}")

    sample_match = re.search(r"Prepared rows:\s*([\d,]+)", cell_text.get(9, ""))
    sample_rows = int(sample_match.group(1).replace(",", "")) if sample_match else int(
        null_summary.iloc[0]["prepared_rows"]
    )

    selected_output_cells = [2, 4, 6, 7, 9, 10, 16, 18, 20, 21, 23, 26]
    execution_outputs = []
    for index in selected_output_cells:
        text = cell_text.get(index, "").strip()
        if not text:
            continue
        text = text.replace(str(ROOT), "<project root>")
        text = re.sub(r"(?m)^Python executable:.*(?:\n|$)", "", text)
        execution_outputs.append((index + 1, text[:1800]))

    source_header = table_for(
        cell_tables.get(6, []), {"column_position", "column_name"}
    )
    if source_header is None:
        raise ValueError("Notebook output is missing the source-header table.")
    source_header = source_header.sort_values("column_position")
    python_match = re.search(r"Python version:\s*([^\s]+)", cell_text.get(2, ""))
    spark_match = re.search(r"Spark version:\s*([^\s]+)", cell_text.get(4, ""))
    master_match = re.search(r"Spark master:\s*([^\s]+)", cell_text.get(4, ""))

    return {
        "notebook": notebook,
        "code_cells": code_cells,
        "cell_text": cell_text,
        "cell_tables": cell_tables,
        "metrics": metrics,
        "class_distribution": class_distribution,
        "null_summary": null_summary,
        "feature_summary": feature_summary,
        "source_header": source_header,
        "python_version": python_match.group(1) if python_match else "not recorded",
        "spark_version": spark_match.group(1) if spark_match else "not recorded",
        "spark_master": master_match.group(1) if master_match else "not recorded",
        "split_summary": split_summary,
        "sql": sql,
        "sample_rows": sample_rows,
        "images": images,
        "execution_outputs": execution_outputs,
    }


def table_rows(frame: pd.DataFrame, limit: int | None = None) -> list[list[str]]:
    selected = frame if limit is None else frame.head(limit)
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rows.append([format_value(value) for value in values])
    return rows


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
    return str(value)


def add_docx_table(
    document: Document,
    headers: list[str],
    rows: list[list[str]],
    note: str | None = None,
) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Light Shading Accent 1"
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = header
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for row in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row):
            cells[index].text = str(value)
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if note:
        paragraph = document.add_paragraph(note)
        paragraph.runs[0].italic = True


def add_docx_frame(document: Document, frame: pd.DataFrame, limit: int | None = None, note: str | None = None) -> None:
    add_docx_table(
        document,
        [str(column) for column in frame.columns],
        table_rows(frame, limit),
        note,
    )


def add_docx_page_numbers(document: Document) -> None:
    for section in document.sections:
        paragraph = section.footer.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        run = paragraph.add_run("Page ")
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        run._r.addnext(field)


def build_docx(evidence: dict[str, Any]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)
    document.styles["Normal"].font.name = "Calibri"
    document.styles["Normal"].font.size = Pt(10)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(
        "Distributed Anti-Money Laundering Transaction Analysis and "
        "Classification Using Apache Spark"
    )
    run.bold = True
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(23, 54, 93)
    subtitle = document.add_paragraph("INT315 – Cluster Computing Project Report")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for label, value in [
        ("Student name", "[Enter student name]"),
        ("Registration number", "[Enter registration number]"),
        ("Section", "[Enter section]"),
        ("University", "[Enter university]"),
        ("Faculty name", "[Enter faculty name]"),
        ("Submission date", "[Enter submission date]"),
    ]:
        document.add_paragraph(f"{label}: {value}")
    note = document.add_paragraph(
        "Educational analysis of simulated AMLSim transactions; not a deployed "
        "banking, real-time monitoring, or automated decision system."
    )
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_page_break()

    document.add_heading("Table of Contents", level=1)
    for title in [
        "1. Project Title",
        "2. Problem Statement",
        "3. Objectives and Scope",
        "4. Dataset Description",
        "5. Dataset Source",
        "6. Tools and Technologies Used",
        "7. Implementation / Methodology",
        "8. Notebook Code",
        "9. Notebook Execution Outputs",
        "10. Output / Result Screenshots",
        "11. Results and Discussion",
        "12. Conclusion and Future Work",
        "References",
    ]:
        document.add_paragraph(title, style="List Bullet")
    document.add_page_break()

    document.add_heading("1. Project Title", level=1)
    document.add_paragraph(
        "Distributed Anti-Money Laundering Transaction Analysis and "
        "Classification Using Apache Spark. Course: INT315 – Cluster Computing."
    )

    document.add_heading("2. Problem Statement", level=1)
    document.add_paragraph(
        "Large transaction data requires repeatable preparation, aggregation, "
        "and evaluation. Rare positive labels make accuracy alone insufficient: "
        "a model may classify nearly every transaction as negative and still "
        "appear accurate. This project demonstrates an offline Spark workflow "
        "on synthetic AMLSim data and compares precision, recall, F1, PR-AUC, "
        "ROC-AUC, and confusion counts."
    )
    document.add_paragraph(
        "AMLSim labels represent simulator-generated activity, not verified "
        "real-world crime. Model scores in this educational notebook are not "
        "proof of laundering and are not recommendations for real-world action."
    )

    document.add_heading("3. Objectives and Scope", level=1)
    for item in [
        "Prepare and validate a reproducible transaction sample using the notebook workflow.",
        "Demonstrate Spark DataFrame transformations and Spark SQL summaries.",
        "Compare Decision Tree, Random Forest, and Gradient-Boosted Tree classifiers.",
        "Use a chronological train/validation/test split and report validation-selected thresholds on the test period.",
        "Display tables, metrics, prediction previews, and plots inline in the notebook.",
    ]:
        document.add_paragraph(item, style="List Number")
    document.add_paragraph(
        "The notebook imports the existing project modules for pipeline operations. "
        "The code reproduced in this report is exclusively the notebook's own "
        "code cells; module files are not copied into this report."
    )

    document.add_heading("4. Dataset Description", level=1)
    document.add_paragraph(
        f"The notebook reports {evidence['sample_rows']:,} prepared transaction "
        "rows and eight prepared columns. The saved class-distribution output is:"
    )
    add_docx_frame(document, evidence["class_distribution"])
    document.add_paragraph(
        "The class imbalance in the notebook output is important when "
        "interpreting threshold-based model results."
    )
    class_counts = {
        str(row["class_name"]): (int(row["count"]), float(row["percentage"]))
        for _, row in evidence["class_distribution"].iterrows()
    }
    document.add_paragraph(
        f"The displayed classes are {class_counts['AML-negative'][0]:,} "
        f"AML-negative rows ({class_counts['AML-negative'][1]:.3f}%) and "
        f"{class_counts['AML-positive'][0]:,} AML-positive rows "
        f"({class_counts['AML-positive'][1]:.3f}%)."
    )
    account_columns = int(
        (evidence["source_header"]["column_name"].astype(str) == "Account").sum()
    )
    document.add_paragraph(
        f"The displayed input header has {len(evidence['source_header'])} columns "
        f"and {account_columns} columns named `Account`; the notebook resolves "
        "these by source position before assigning sender and receiver fields."
    )

    document.add_heading("5. Dataset Source", level=1)
    document.add_paragraph(
        "The notebook identifies the input as IBM AMLSim HI-Small transaction "
        "data. Its saved execution output states that the source CSV and the "
        "500,000-row sample were available, and that the existing sample was "
        "reused rather than resampled for this run. Source-row counts were not "
        "recalculated in that notebook run."
    )
    document.add_paragraph(
        "References: IBM AMLSim, https://github.com/IBM/AMLSim; IBM AMLSim "
        "example data wiki, https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set."
    )

    document.add_heading("6. Tools and Technologies Used", level=1)
    add_docx_table(
        document,
        ["Technology", "Role shown in the notebook"],
        [
            [f"Python {evidence['python_version']}", "Notebook orchestration and analysis code."],
            [f"Apache Spark / PySpark {evidence['spark_version']}", "Local Spark DataFrames, SQL, and Spark ML."],
            [f"Spark {evidence['spark_master']}", "Two local worker threads on one machine."],
            ["Pandas / NumPy", "Compact metrics, table presentation, and analysis."],
            ["Matplotlib / Seaborn", "Inline notebook visualizations."],
            ["Jupyter", "Combined code cells, narrative, tabular outputs, and plots."],
        ],
    )
    document.add_paragraph(
        f"The notebook records Python {evidence['python_version']}, Spark "
        f"{evidence['spark_version']}, and master `{evidence['spark_master']}`. "
        "This is local execution, not evidence of a multi-machine cluster."
    )

    document.add_heading("7. Implementation / Methodology", level=1)
    for title, description in [
        ("7.1 Notebook setup", "The notebook locates the project, imports its pipeline modules, and exposes run controls for sampling, training, and SQL queries."),
        ("7.2 Loading and preparation", "The notebook displays the source header, prepared schema, sample count, a small record preview, and a null-count summary."),
        ("7.3 Sample profile", "Class distribution and amount-range summaries are calculated and visualized inline."),
        ("7.4 SQL analysis", "Seven Spark SQL queries display high-value transactions, active senders, payment types, transaction statistics, a sender-summary join, a wildcard search, and AML label rates by payment type."),
        ("7.5 Modeling and evaluation", "Three classifiers are evaluated with a chronological split. Metrics include threshold-based precision, recall, F1, confusion counts, and score-based PR-AUC/ROC-AUC."),
        ("7.6 Output handling", "Notebook tables and figures are rendered inline. The final notebook cell releases the cached transactions and stops the Spark session."),
    ]:
        document.add_heading(title, level=2)
        document.add_paragraph(description)
    document.add_paragraph(
        "The split counts displayed in the notebook are included in Section 11. "
        "The output tables are stored in the notebook file; the report generator "
        "does not run Spark or read result CSVs and chart artifacts."
    )

    document.add_heading("8. Notebook Code", level=1)
    document.add_paragraph(
        "The following source is taken only from non-empty code cells in "
        f"`{NOTEBOOK_PATH.relative_to(ROOT)}`. The notebook delegates pipeline "
        "logic to project imports; this report intentionally does not append "
        "standalone Python modules."
    )
    for cell_number, cell in evidence["code_cells"]:
        source = "".join(cell.get("source", [])).rstrip()
        document.add_heading(f"Notebook code cell {cell_number + 1}", level=2)
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.line_spacing = 1.0
        run = paragraph.add_run(source)
        run.font.name = "Consolas"
        run.font.size = Pt(7)
        run.font.color.rgb = RGBColor(35, 35, 35)

    document.add_heading("9. Notebook Execution Outputs", level=1)
    document.add_paragraph(
        "These are text outputs embedded in the notebook, not recreated "
        "terminal screenshots. They document the notebook's saved execution "
        "state; this report does not rerun the Spark workflow."
    )
    for cell_number, text in evidence["execution_outputs"]:
        document.add_heading(f"Output from notebook cell {cell_number}", level=2)
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(text)
        run.font.name = "Consolas"
        run.font.size = Pt(8)

    document.add_heading("10. Output / Result Screenshots", level=1)
    document.add_paragraph(
        "The figures below are actual inline PNG outputs embedded in the "
        "notebook. They are notebook charts, not screenshots of a desktop "
        "application, and they are not loaded from the standalone artifacts folder."
    )
    for index, image in enumerate(evidence["images"], start=1):
        document.add_picture(str(image["path"]), width=Inches(6.0))
        document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = document.add_paragraph(f"Figure {index}. {image['caption']}")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption.runs:
            caption.runs[0].italic = True

    document.add_heading("Notebook output tables", level=2)
    document.add_heading("Prepared-data null summary", level=3)
    add_docx_frame(document, evidence["null_summary"])
    if evidence["feature_summary"] is not None:
        document.add_heading("Amount-range feature summary", level=3)
        add_docx_frame(document, evidence["feature_summary"])
    document.add_heading("SQL output sizes", level=3)
    add_docx_table(
        document,
        ["Notebook SQL output", "Rows displayed"],
        [[name, str(len(frame))] for name, frame in evidence["sql"].items()],
    )
    add_docx_frame(
        document,
        evidence["sql"]["aml_rate_by_payment_type"],
        note="Label rates describe this synthetic sample; they are not causal risk estimates.",
    )
    for name in [
        "high_value_transactions",
        "active_senders",
        "payment_types",
        "transaction_statistics",
        "sender_account_join",
        "wildcard_payment_search",
    ]:
        document.add_heading(name.replace("_", " ").title(), level=3)
        add_docx_frame(
            document,
            evidence["sql"][name],
            limit=5,
            note="First five rows of the saved notebook display; the complete output remains in the notebook.",
        )

    document.add_heading("11. Results and Discussion", level=1)
    document.add_paragraph(
        "The following results are read from the notebook's embedded metrics "
        "and split tables. Threshold-dependent measures use each classifier's "
        "validation-selected threshold; PR-AUC and ROC-AUC use continuous "
        "test-set scores."
    )
    metric_display = evidence["metrics"].copy()
    metric_display["model"] = metric_display["model"].replace(
        {
            "DecisionTreeClassifier": "Decision Tree",
            "RandomForestClassifier": "Random Forest",
            "GBTClassifier": "Gradient-Boosted Trees",
        }
    )
    for column in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"):
        metric_display[column] = metric_display[column].map(lambda value: f"{value:.4f}")
    add_docx_frame(document, metric_display)
    document.add_heading("Chronological split counts", level=2)
    add_docx_frame(document, evidence["split_summary"])
    document.add_heading("Confusion counts", level=2)
    add_docx_table(
        document,
        ["Model", "TP", "TN", "FP", "FN"],
        [
            [
                row["model"],
                str(row["tp"]),
                str(row["tn"]),
                str(row["fp"]),
                str(row["fn"]),
            ]
            for _, row in evidence["metrics"].iterrows()
        ],
    )
    best_f1 = evidence["metrics"].loc[evidence["metrics"]["f1"].idxmax()]
    best_recall = evidence["metrics"].loc[evidence["metrics"]["recall"].idxmax()]
    best_pr_auc = evidence["metrics"].loc[evidence["metrics"]["pr_auc"].idxmax()]
    best_roc_auc = evidence["metrics"].loc[evidence["metrics"]["roc_auc"].idxmax()]
    document.add_paragraph(
        f"In the saved notebook output, {best_f1['model']} has the highest F1 "
        f"({best_f1['f1']:.4f}); {best_recall['model']} has the highest recall "
        f"({best_recall['recall']:.4f}); {best_pr_auc['model']} has the highest "
        f"PR-AUC ({best_pr_auc['pr_auc']:.4f}); and {best_roc_auc['model']} "
        f"has the highest ROC-AUC ({best_roc_auc['roc_auc']:.4f})."
    )
    document.add_paragraph(
        "The Decision Tree's high recall is accompanied by many false positives. "
        "The Random Forest has the strongest F1 and ranking scores in this run "
        "but lower recall. Positive prevalence is low, so high accuracy or "
        "ROC-AUC alone does not demonstrate useful real-world AML detection."
    )

    document.add_heading("12. Conclusion and Future Work", level=1)
    document.add_paragraph(
        "The notebook demonstrates an end-to-end local Spark workflow using "
        "synthetic AMLSim data, inline SQL results, three classifiers, and "
        "imbalance-aware evaluation. Results are limited by the synthetic "
        "labels, notebook's selected features, one sampled dataset, and one "
        "chronological test period."
    )
    document.add_paragraph(
        "Future educational work could compare additional temporal periods, "
        "richer account/network features, calibration, and threshold policies "
        "tied to a defined review capacity. Any real-world use would require "
        "governed data, independent validation, compliance controls, and "
        "human oversight; none is implemented here."
    )

    document.add_heading("References", level=1)
    for reference in [
        "[1] IBM Research, AMLSim: Anti-Money Laundering Simulator. https://github.com/IBM/AMLSim",
        "[2] IBM AMLSim Wiki, Download Example Data Set. https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set",
        "[3] Apache Software Foundation, Spark 3.5 Documentation. https://spark.apache.org/docs/3.5.9/",
        "[4] Fintech-secure-aml-spark project notebook, notebooks/01_aml_spark_workflow.ipynb.",
    ]:
        document.add_paragraph(reference, style="List Number")

    add_docx_page_numbers(document)
    REPORTS.mkdir(parents=True, exist_ok=True)
    document.save(DOCX_PATH)


def build_pdf(evidence: dict[str, Any]) -> None:
    font_dir = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("ReportSans", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("ReportMono", str(font_dir / "DejaVuSansMono.ttf")))
    base = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle(
            "ReportBody", parent=base["BodyText"], fontName="ReportSans",
            fontSize=8.5, leading=11, spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "ReportSmall", parent=base["BodyText"], fontName="ReportSans",
            fontSize=7, leading=8.5,
        ),
        "h1": ParagraphStyle(
            "ReportH1", parent=base["Heading1"], fontName="ReportSans-Bold",
            fontSize=14, leading=17, textColor=colors.HexColor("#17365D"),
            spaceBefore=10, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=base["Heading2"], fontName="ReportSans-Bold",
            fontSize=10, leading=13, textColor=colors.HexColor("#2F5597"),
            spaceBefore=7, spaceAfter=4,
        ),
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="ReportSans-Bold",
            fontSize=21, leading=26, textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER, spaceAfter=14,
        ),
        "center": ParagraphStyle(
            "ReportCenter", parent=base["Normal"], fontName="ReportSans",
            fontSize=10, leading=13, alignment=TA_CENTER,
        ),
        "caption": ParagraphStyle(
            "ReportCaption", parent=base["BodyText"], fontName="ReportSans",
            fontSize=7.5, leading=9, alignment=TA_CENTER, spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "ReportCode", parent=base["Code"], fontName="ReportMono",
            fontSize=6.2, leading=7.2, spaceAfter=0,
        ),
    }
    story: list[Any] = []

    def paragraph(text: str, style: str = "body") -> None:
        story.append(Paragraph(text, styles[style]))

    def heading(text: str, level: int = 1) -> None:
        paragraph(escape(text), "h1" if level == 1 else "h2")

    def report_table(headers: list[str], rows: list[list[str]]) -> None:
        data = [[Paragraph(escape(value), styles["small"]) for value in headers]]
        data.extend(
            [[Paragraph(escape(str(value)), styles["small"]) for value in row] for row in rows]
        )
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C9D6")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.extend([table, Spacer(1, 6)])

    def dataframe_table(frame: pd.DataFrame, limit: int | None = None) -> None:
        report_table(
            [str(column) for column in frame.columns],
            table_rows(frame, limit),
        )

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#B7C9D6"))
        canvas.line(19 * mm, 14 * mm, A4[0] - 19 * mm, 14 * mm)
        canvas.setFont("ReportSans", 7)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(19 * mm, 9 * mm, "INT315 | Cluster Computing")
        canvas.drawRightString(A4[0] - 19 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    story.extend([Spacer(1, 35 * mm)])
    paragraph(
        "Distributed Anti-Money Laundering Transaction Analysis and "
        "Classification Using Apache Spark",
        "title",
    )
    paragraph("INT315 – Cluster Computing Project Report", "center")
    story.append(Spacer(1, 15 * mm))
    report_table(
        ["Cover information", "Value"],
        [
            ["Student name", "[Enter student name]"],
            ["Registration number", "[Enter registration number]"],
            ["Section", "[Enter section]"],
            ["University", "[Enter university]"],
            ["Faculty name", "[Enter faculty name]"],
            ["Submission date", "[Enter submission date]"],
        ],
    )
    story.append(Spacer(1, 10 * mm))
    paragraph(
        "Educational analysis of simulated AMLSim transactions; not a deployed "
        "banking, real-time monitoring, or automated decision system.",
        "caption",
    )
    story.append(PageBreak())

    heading("Table of Contents")
    for name in [
        "1. Project Title",
        "2. Problem Statement",
        "3. Objectives and Scope",
        "4. Dataset Description",
        "5. Dataset Source",
        "6. Tools and Technologies Used",
        "7. Implementation / Methodology",
        "8. Notebook Code",
        "9. Notebook Execution Outputs",
        "10. Output / Result Screenshots",
        "11. Results and Discussion",
        "12. Conclusion and Future Work",
        "References",
    ]:
        paragraph(f"&bull;&nbsp;&nbsp;{escape(name)}")
    story.append(PageBreak())

    heading("1. Project Title")
    paragraph(
        "Distributed Anti-Money Laundering Transaction Analysis and "
        "Classification Using Apache Spark. Course: INT315 – Cluster Computing."
    )
    heading("2. Problem Statement")
    paragraph(
        "Large transaction data requires repeatable preparation, aggregation, "
        "and evaluation. Rare positive labels make accuracy alone insufficient. "
        "This project demonstrates an offline Spark workflow on synthetic AMLSim "
        "data and compares precision, recall, F1, PR-AUC, ROC-AUC, and confusion counts."
    )
    paragraph(
        "AMLSim labels represent simulator-generated activity, not verified "
        "real-world crime. Notebook scores are not proof of laundering or "
        "recommendations for real-world action."
    )
    heading("3. Objectives and Scope")
    for item in [
        "Prepare and validate a reproducible transaction sample.",
        "Demonstrate Spark DataFrame transformations and Spark SQL summaries.",
        "Compare Decision Tree, Random Forest, and Gradient-Boosted Tree classifiers.",
        "Use a chronological train/validation/test split and report test results.",
        "Display tables, metrics, previews, and plots inline in the notebook.",
    ]:
        paragraph(f"&bull;&nbsp;&nbsp;{escape(item)}")
    paragraph(
        "The notebook imports existing project modules for pipeline operations. "
        "Only notebook code cells are reproduced here; standalone modules are not appended."
    )
    heading("4. Dataset Description")
    paragraph(
        f"The notebook reports {evidence['sample_rows']:,} prepared transaction rows "
        "and eight prepared columns. The saved class distribution is:"
    )
    dataframe_table(evidence["class_distribution"])
    paragraph(
        "Class imbalance is visible in the saved table. The input header has "
        f"{len(evidence['source_header'])} columns, including "
        f"{int((evidence['source_header']['column_name'].astype(str) == 'Account').sum())} "
        "source columns named Account that are resolved by position."
    )
    heading("5. Dataset Source")
    paragraph(
        "The notebook identifies IBM AMLSim HI-Small transaction data. Its "
        "saved output states that the source CSV and the existing 500,000-row "
        "sample were available; the sample was reused and source-row counts "
        "were not recalculated in this run."
    )
    paragraph(
        'IBM AMLSim: <link href="https://github.com/IBM/AMLSim" color="blue">github.com/IBM/AMLSim</link>; '
        'example data: <link href="https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set" color="blue">AMLSim Wiki</link>.'
    )
    heading("6. Tools and Technologies Used")
    report_table(
        ["Technology", "Role shown in the notebook"],
        [
            [f"Python {evidence['python_version']}", "Notebook orchestration and analysis code."],
            [f"Apache Spark / PySpark {evidence['spark_version']}", "Local Spark DataFrames, SQL, and Spark ML."],
            [f"Spark {evidence['spark_master']}", "Two local worker threads on one machine."],
            ["Pandas / NumPy", "Compact metrics, tables, and analysis."],
            ["Matplotlib / Seaborn", "Inline notebook visualizations."],
            ["Jupyter", "Code cells, narrative, tabular outputs, and plots."],
        ],
    )
    paragraph(
        f"The notebook records Python {escape(evidence['python_version'])}, "
        f"Spark {escape(evidence['spark_version'])}, and master "
        f"{escape(evidence['spark_master'])}."
    )

    heading("7. Implementation / Methodology")
    for title, description in [
        ("7.1 Notebook setup", "The notebook locates the project, imports its modules, and provides controls for sampling, training, and SQL queries."),
        ("7.2 Loading and preparation", "It displays the source header, prepared schema, row count, sample preview, and null summary."),
        ("7.3 Sample profile", "Class distribution and amount-range summaries are calculated and plotted inline."),
        ("7.4 SQL analysis", "Seven queries cover transaction values, active senders, payment types, aggregate statistics, a sender join, wildcard search, and AML rates by payment type."),
        ("7.5 Modeling and evaluation", "Three classifiers are evaluated chronologically. Threshold-based metrics and continuous-score ranking metrics are both reported."),
        ("7.6 Output handling", "Notebook tables and figures are inline; the final cell releases cached data and stops Spark."),
    ]:
        heading(title, 2)
        paragraph(description)
    paragraph(
        "The report generator reads embedded notebook outputs only. It does not "
        "execute Spark or load model metrics, SQL CSVs, or charts from the artifacts directory."
    )

    heading("8. Notebook Code")
    paragraph(
        f"Source below is taken only from non-empty code cells in "
        f"{escape(str(NOTEBOOK_PATH.relative_to(ROOT)))}. Pipeline logic imported "
        "from other project files is not copied into this report."
    )
    for cell_number, cell in evidence["code_cells"]:
        source = "".join(cell.get("source", [])).rstrip()
        heading(f"Notebook code cell {cell_number + 1}", 2)
        story.append(Preformatted(source, styles["code"], maxLineLength=110))

    heading("9. Notebook Execution Outputs")
    paragraph(
        "These are text outputs embedded in the notebook, not recreated terminal "
        "screenshots. This report does not rerun the workflow."
    )
    for cell_number, text in evidence["execution_outputs"]:
        heading(f"Output from notebook cell {cell_number}", 2)
        story.append(Preformatted(text, styles["code"], maxLineLength=110))

    heading("10. Output / Result Screenshots")
    paragraph(
        "The figures below are actual inline PNG outputs embedded in the notebook. "
        "They are not desktop screenshots and are not loaded from the artifacts folder."
    )
    for index, image in enumerate(evidence["images"], start=1):
        item = PdfImage(str(image["path"]))
        max_width, max_height = 172 * mm, 95 * mm
        scale = min(max_width / item.imageWidth, max_height / item.imageHeight)
        item.drawWidth = item.imageWidth * scale
        item.drawHeight = item.imageHeight * scale
        item.hAlign = "CENTER"
        story.extend([item, Spacer(1, 4)])
        paragraph(f"Figure {index}. {escape(image['caption'])}", "caption")

    heading("Notebook output tables", 2)
    heading("Prepared-data null summary", 2)
    dataframe_table(evidence["null_summary"])
    if evidence["feature_summary"] is not None:
        heading("Amount-range feature summary", 2)
        dataframe_table(evidence["feature_summary"])
    heading("SQL output sizes", 2)
    report_table(
        ["Notebook SQL output", "Rows displayed"],
        [[name, str(len(frame))] for name, frame in evidence["sql"].items()],
    )
    heading("AML rate by payment type", 2)
    dataframe_table(evidence["sql"]["aml_rate_by_payment_type"])
    for name in [
        "high_value_transactions",
        "active_senders",
        "payment_types",
        "transaction_statistics",
        "sender_account_join",
        "wildcard_payment_search",
    ]:
        heading(name.replace("_", " ").title(), 2)
        dataframe_table(evidence["sql"][name].head(5))
        paragraph("First five rows of the saved notebook display; the full table remains in the notebook.")

    heading("11. Results and Discussion")
    paragraph(
        "Values below are taken from the notebook's embedded metrics and split "
        "tables. Threshold-based measures use validation-selected cutoffs; "
        "PR-AUC and ROC-AUC use continuous test-set scores."
    )
    metric_display = evidence["metrics"].copy()
    metric_display["model"] = metric_display["model"].replace(
        {
            "DecisionTreeClassifier": "Decision Tree",
            "RandomForestClassifier": "Random Forest",
            "GBTClassifier": "Gradient-Boosted Trees",
        }
    )
    for column in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc", "threshold"):
        metric_display[column] = metric_display[column].map(lambda value: f"{value:.4f}")
    dataframe_table(metric_display)
    heading("Chronological split counts", 2)
    dataframe_table(evidence["split_summary"])
    heading("Confusion counts", 2)
    report_table(
        ["Model", "TP", "TN", "FP", "FN"],
        [
            [row["model"], str(row["tp"]), str(row["tn"]), str(row["fp"]), str(row["fn"])]
            for _, row in evidence["metrics"].iterrows()
        ],
    )
    best_f1 = evidence["metrics"].loc[evidence["metrics"]["f1"].idxmax()]
    best_recall = evidence["metrics"].loc[evidence["metrics"]["recall"].idxmax()]
    best_pr_auc = evidence["metrics"].loc[evidence["metrics"]["pr_auc"].idxmax()]
    best_roc_auc = evidence["metrics"].loc[evidence["metrics"]["roc_auc"].idxmax()]
    paragraph(
        f"In the saved notebook output, {escape(str(best_f1['model']))} has the "
        f"highest F1 ({best_f1['f1']:.4f}); {escape(str(best_recall['model']))} "
        f"has the highest recall ({best_recall['recall']:.4f}); "
        f"{escape(str(best_pr_auc['model']))} has the highest PR-AUC "
        f"({best_pr_auc['pr_auc']:.4f}); and {escape(str(best_roc_auc['model']))} "
        f"has the highest ROC-AUC ({best_roc_auc['roc_auc']:.4f})."
    )
    paragraph(
        "The Decision Tree's high recall is accompanied by many false positives. "
        "The Random Forest has the strongest F1 and ranking scores in this run "
        "but lower recall. Positive prevalence is low, so accuracy or ROC-AUC "
        "alone does not demonstrate useful real-world AML detection."
    )
    heading("12. Conclusion and Future Work")
    paragraph(
        "The notebook demonstrates a local Spark workflow using synthetic AMLSim "
        "data, inline SQL results, three classifiers, and imbalance-aware evaluation. "
        "Results are limited by synthetic labels, selected features, one sampled "
        "dataset, and one chronological test period."
    )
    paragraph(
        "Future educational work could compare additional temporal periods, "
        "richer account/network features, calibration, and thresholds tied to a "
        "defined review capacity. Real-world use would require governed data, "
        "independent validation, compliance controls, and human oversight; none "
        "is implemented here."
    )
    heading("References")
    paragraph(
        '[1] IBM AMLSim. <link href="https://github.com/IBM/AMLSim" color="blue">github.com/IBM/AMLSim</link>.',
        "small",
    )
    paragraph(
        '[2] IBM AMLSim Wiki. <link href="https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set" color="blue">Download Example Data Set</link>.',
        "small",
    )
    paragraph(
        '[3] Apache Spark 3.5.9 Documentation. <link href="https://spark.apache.org/docs/3.5.9/" color="blue">spark.apache.org/docs/3.5.9</link>.',
        "small",
    )
    paragraph("[4] Project notebook: notebooks/01_aml_spark_workflow.ipynb.", "small")

    pdf = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="INT315 Cluster Computing Project Report",
        author="Student name to be completed",
    )
    pdf.build(story, onFirstPage=footer, onLaterPages=footer)


def validate_reports(evidence: dict[str, Any]) -> None:
    doc = Document(DOCX_PATH)
    doc_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    required_sections = [
        "1. Project Title",
        "2. Problem Statement",
        "3. Objectives and Scope",
        "4. Dataset Description",
        "5. Dataset Source",
        "6. Tools and Technologies Used",
        "7. Implementation / Methodology",
        "8. Notebook Code",
        "9. Notebook Execution Outputs",
        "10. Output / Result Screenshots",
        "11. Results and Discussion",
        "12. Conclusion and Future Work",
    ]
    for title in required_sections:
        if title not in doc_text:
            raise RuntimeError(f"DOCX is missing section: {title}")
    if len(doc.inline_shapes) != len(evidence["images"]):
        raise RuntimeError("DOCX notebook figure count does not match embedded notebook images.")
    for _, cell in evidence["code_cells"]:
        source = "".join(cell.get("source", [])).strip()
        if source and source not in doc_text:
            raise RuntimeError("DOCX is missing a notebook code cell.")

    reader = PdfReader(str(PDF_PATH))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    for title in required_sections:
        if title not in pdf_text:
            raise RuntimeError(f"PDF is missing section: {title}")
    for _, row in evidence["metrics"].iterrows():
        if f"{row['f1']:.4f}" not in pdf_text:
            raise RuntimeError(f"PDF is missing notebook metric for {row['model']}.")
    if "Appendix A — main.py" in doc_text or "artifacts/hi-small-500k/" in doc_text:
        raise RuntimeError("Report contains standalone-source or saved-artifact content.")
    print(f"DOCX verified: {DOCX_PATH} ({DOCX_PATH.stat().st_size:,} bytes)")
    print(f"PDF verified: {PDF_PATH} ({len(reader.pages)} pages, {PDF_PATH.stat().st_size:,} bytes)")
    print(
        f"Notebook-only code and {len(evidence['images'])} embedded images verified; "
        f"{len(evidence['metrics'])} metric rows checked."
    )


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="int315-notebook-report-") as temporary_directory:
        evidence = load_notebook_evidence(Path(temporary_directory))
        build_docx(evidence)
        build_pdf(evidence)
        validate_reports(evidence)


if __name__ == "__main__":
    main()
