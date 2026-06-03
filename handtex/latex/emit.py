"""Emit LaTeX from a laid-out :class:`Document`.

Super/subscripts become ``^{...}`` / ``_{...}``; symbol names are mapped to
their LaTeX commands via the symbol table. The output is math-mode source by
default (most whiteboard content is math); :func:`standalone_document` wraps
it into a compilable file, optionally typeset in your handwritten font.
"""

from __future__ import annotations

from typing import List

from ..types import Document, Line, Token
from ..font import symbols


def _token_latex(tok: Token) -> str:
    return symbols.latex_for(tok.name)


def _group_latex(tokens: List[Token]) -> str:
    parts = [_token_latex(t) for t in tokens]
    return "".join(parts)


def _emit_token(tok: Token) -> str:
    out = _token_latex(tok)
    if tok.superscript:
        out += "^{" + _group_latex(tok.superscript) + "}"
    if tok.subscript:
        out += "_{" + _group_latex(tok.subscript) + "}"
    return out


def _emit_line(line: Line) -> str:
    return " ".join(_emit_token(t) for t in line.tokens)


def document_to_latex(doc: Document, math_mode: bool = True) -> str:
    """Render the document as a LaTeX string.

    Lines are separated by ``\\\\``. When ``math_mode`` is True the result is
    meant to sit inside ``$...$`` / an equation environment.
    """
    body = " \\\\\n".join(_emit_line(line) for line in doc.lines)
    if math_mode:
        return body
    return body


_STANDALONE_TMPL = r"""\documentclass[border=10pt]{{standalone}}
{font_setup}\begin{{document}}
$\displaystyle
{body}
$
\end{{document}}
"""

_FONT_SETUP_XETEX = r"""\usepackage{{fontspec}}
\usepackage{{unicode-math}}
\setmathfont{{{font_path}}}
\setmainfont{{{font_path}}}
"""


def standalone_document(doc: Document, font_path: str | None = None) -> str:
    """Wrap the document into a compilable standalone .tex file.

    If ``font_path`` is given, the file is set up for XeLaTeX/LuaLaTeX so the
    math is typeset in your handwritten font via ``unicode-math`` — the same
    .ttf you'd drop into any other application.
    """
    font_setup = ""
    if font_path:
        font_setup = _FONT_SETUP_XETEX.format(font_path=font_path)
    body = document_to_latex(doc, math_mode=True)
    return _STANDALONE_TMPL.format(font_setup=font_setup, body=body)
