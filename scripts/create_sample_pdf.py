"""Generate a sample medical PDF for testing ingestion and Q&A."""

from pathlib import Path

from fpdf import FPDF

SAMPLE_TEXT = """
Medical Reference: Asthma and Hypertension (Sample)

ASTHMA
Asthma is a chronic respiratory condition characterized by inflammation and narrowing
of the airways. Common causes include genetic predisposition, environmental allergens
(pollen, dust mites), air pollution, respiratory infections, and occupational irritants.
Symptoms include wheezing, shortness of breath, chest tightness, and coughing.
Management includes inhaled bronchodilators, corticosteroids, and trigger avoidance.

HYPERTENSION
Hypertension (high blood pressure) is defined as sustained blood pressure at or above
140/90 mmHg. Primary causes include genetics, high sodium intake, obesity, sedentary
lifestyle, chronic stress, and excessive alcohol use. Secondary hypertension may result
from kidney disease, endocrine disorders, or certain medications.
Treatment includes lifestyle modification, ACE inhibitors, ARBs, diuretics, and
regular monitoring.

DIABETES TYPE 2
Type 2 diabetes involves insulin resistance and relative insulin deficiency. Risk factors
include obesity, family history, physical inactivity, and age over 45. Complications
can affect cardiovascular, renal, and nervous systems. First-line management includes
metformin, diet, and exercise.
"""


def create_sample_pdf(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sample_medical_reference.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in SAMPLE_TEXT.strip().split("\n"):
        pdf.multi_cell(0, 6, line.strip())
        pdf.ln(1)

    pdf.output(str(out_path))
    print(f"Created: {out_path}")
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    create_sample_pdf(root / "data" / "medical_pdfs")
