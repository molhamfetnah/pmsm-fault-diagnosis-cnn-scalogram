"""Replace exotic Unicode symbols with ASCII/word equivalents so the Arabic PDF
build (xelatex + Amiri, which lacks math/arrow glyphs) renders cleanly. Used only
for the PDF path; the DOCX/PPTX keep the original symbols (Word/PowerPoint render
them via system font fallback). Usage: python _sanitize_for_pdf.py in.md out.md
"""
import sys

REPLACEMENTS = {
    "⟵": "<-", "⟶": "->", "→": "->", "←": "<-", "↔": "<->",
    "≫": ">>", "≪": "<<", "≥": ">=", "≤": "<=", "≈": "~=", "≠": "!=",
    "×": "x", "·": ".", "÷": "/",
    "²": "^2", "³": "^3",
    "Δ": "Delta", "δ": "delta", "ω": "omega", "Ω": "Omega",
    "α": "alpha", "β": "beta", "σ": "sigma", "μ": "mu", "θ": "theta",
    "ψ": "psi", "φ": "phi", "π": "pi", "λ": "lambda", "τ": "tau",
    "∫": "∮".replace("∮", "S"), "∞": "inf", "∂": "d", "√": "sqrt", "∑": "sum",
    "−": "-", "₀": "0", "₁": "1", "₂": "2", "₃": "3", "ₐ": "a", "ᵢ": "i", "ₖ": "k",
    "…": "...", "—": "-", "–": "-", "►": ">", "─": "-", "✓": "[x]",
}


def sanitize(text):
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    return text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        out = sanitize(f.read())
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"sanitized {src} -> {dst}")
