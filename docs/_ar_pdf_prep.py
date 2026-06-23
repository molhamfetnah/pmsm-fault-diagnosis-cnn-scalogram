"""Prepare an Arabic Markdown file for the xelatex PDF build:
  1) sanitize exotic glyphs (reuse _sanitize_for_pdf.REPLACEMENTS), and
  2) replace every `![caption](path)` image with a raw-LaTeX block that forces the
     image LEFT-TO-RIGHT (\\LR) and an explicit width — otherwise RTL/bidi flushes
     figures off the page.

Usage: python _ar_pdf_prep.py in.md out.md [--beamer]
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from _sanitize_for_pdf import sanitize  # noqa: E402

IMG = re.compile(r'^!\[(?P<cap>.*?)\]\((?P<path>[^)]+?)\)\s*$', re.MULTILINE)


def wrap_article(m):
    cap, path = m.group("cap"), m.group("path")
    # \centerline centers a box in the column regardless of text direction (more
    # robust than figure+\centering under RTL/bidi); caption rendered as text.
    return ("```{=latex}\n"
            "\\par\\medskip\n"
            f"\\centerline{{\\includegraphics[width=0.82\\linewidth,keepaspectratio]{{{path}}}}}\n"
            "\\par\\smallskip\n"
            f"{{\\centering\\small\\itshape {cap}\\par}}\n"
            "\\medskip\n"
            "```")


def wrap_beamer(m):
    path = m.group("path")
    return ("```{=latex}\n"
            f"\\centerline{{\\includegraphics[width=0.82\\linewidth,height=0.62\\textheight,keepaspectratio]{{{path}}}}}\n"
            "```")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    beamer = "--beamer" in sys.argv[3:]
    text = sanitize(open(src, encoding="utf-8").read())
    text = IMG.sub(wrap_beamer if beamer else wrap_article, text)
    open(dst, "w", encoding="utf-8").write(text)
    print(f"prepped {src} -> {dst}{' (beamer)' if beamer else ''}")
