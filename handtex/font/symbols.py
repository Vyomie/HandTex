"""The symbol table — the heart of "works as a font AND in LaTeX".

Every glyph HandTex knows about is one :class:`Symbol` with:

  * ``name``      canonical id used throughout the pipeline ("alpha")
  * ``char``      the actual Unicode character ("α")
  * ``codepoint`` Unicode code point (0x03B1)
  * ``latex``     the LaTeX command ("\\alpha"), or the literal for ASCII
  * ``vhint``     intrinsic vertical placement hint for punctuation-like
                  marks, used by the layout engine to disambiguate e.g.
                  a comma (low) from an apostrophe (high).

Because each glyph is bound to a real Unicode code point, the generated
font renders the math symbol when you type that code point in *any*
application, and LaTeX (XeLaTeX/LuaLaTeX + fontspec/unicode-math) picks up
the very same outlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional


class VHint(str, Enum):
    """Intrinsic vertical placement of a mark within the line."""

    NORMAL = "normal"   # sits on the baseline like a letter
    HIGH = "high"       # rides high: ' " * ` degree, etc.
    LOW = "low"         # hangs low: , . _ etc.


@dataclass(frozen=True)
class Symbol:
    name: str
    char: str
    latex: str
    vhint: VHint = VHint.NORMAL

    @property
    def codepoint(self) -> int:
        return ord(self.char)

    @property
    def glyph_name(self) -> str:
        """A font-table-safe glyph name."""
        return _safe_glyph_name(self.name, self.char)


def _safe_glyph_name(name: str, char: str) -> str:
    # ASCII letters/digits map to conventional glyph names; everything else
    # gets a uniXXXX name so the font stays portable.
    if char.isascii() and (char.isalnum()):
        return char if char.isalpha() else _DIGIT_NAMES.get(char, f"uni{ord(char):04X}")
    return f"uni{ord(char):04X}"


_DIGIT_NAMES = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _build_table() -> List[Symbol]:
    syms: List[Symbol] = []

    # --- ASCII letters & digits -------------------------------------------
    for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        syms.append(Symbol(name=c, char=c, latex=c))
    for c in "0123456789":
        digit_names = {
            "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        }
        syms.append(Symbol(name=digit_names[c], char=c, latex=c))

    # --- ASCII punctuation / operators ------------------------------------
    # (name, char, latex, vhint)
    ascii_punct = [
        ("space", " ", " ", VHint.NORMAL),
        ("plus", "+", "+", VHint.NORMAL),
        ("minus", "-", "-", VHint.NORMAL),
        ("equals", "=", "=", VHint.NORMAL),
        ("slash", "/", "/", VHint.NORMAL),
        ("lparen", "(", "(", VHint.NORMAL),
        ("rparen", ")", ")", VHint.NORMAL),
        ("lbracket", "[", "[", VHint.NORMAL),
        ("rbracket", "]", "]", VHint.NORMAL),
        ("less", "<", "<", VHint.NORMAL),
        ("greater", ">", ">", VHint.NORMAL),
        ("bar", "|", "|", VHint.NORMAL),
        ("exclam", "!", "!", VHint.NORMAL),
        ("question", "?", "?", VHint.NORMAL),
        ("colon", ":", ":", VHint.NORMAL),
        ("semicolon", ";", ";", VHint.NORMAL),
        # placement-sensitive marks:
        ("comma", ",", ",", VHint.LOW),
        ("period", ".", ".", VHint.LOW),
        ("underscore", "_", "\\_", VHint.LOW),
        ("apostrophe", "'", "'", VHint.HIGH),
        ("quotedbl", '"', "\"", VHint.HIGH),
        ("asterisk", "*", "*", VHint.HIGH),
        ("grave", "`", "`", VHint.HIGH),
        ("prime", "′", "\\prime", VHint.HIGH),
        ("degree", "°", "^\\circ", VHint.HIGH),
    ]
    for name, char, latex, vhint in ascii_punct:
        syms.append(Symbol(name=name, char=char, latex=latex, vhint=vhint))

    # --- Greek (lower & upper) --------------------------------------------
    greek = [
        ("alpha", "α", "\\alpha"), ("beta", "β", "\\beta"),
        ("gamma", "γ", "\\gamma"), ("delta", "δ", "\\delta"),
        ("epsilon", "ε", "\\epsilon"), ("zeta", "ζ", "\\zeta"),
        ("eta", "η", "\\eta"), ("theta", "θ", "\\theta"),
        ("iota", "ι", "\\iota"), ("kappa", "κ", "\\kappa"),
        ("lambda", "λ", "\\lambda"), ("mu", "μ", "\\mu"),
        ("nu", "ν", "\\nu"), ("xi", "ξ", "\\xi"),
        ("pi", "π", "\\pi"), ("rho", "ρ", "\\rho"),
        ("sigma", "σ", "\\sigma"), ("tau", "τ", "\\tau"),
        ("phi", "φ", "\\phi"), ("chi", "χ", "\\chi"),
        ("psi", "ψ", "\\psi"), ("omega", "ω", "\\omega"),
        ("Gamma", "Γ", "\\Gamma"), ("Delta", "Δ", "\\Delta"),
        ("Theta", "Θ", "\\Theta"), ("Lambda", "Λ", "\\Lambda"),
        ("Xi", "Ξ", "\\Xi"), ("Pi", "Π", "\\Pi"),
        ("Sigma", "Σ", "\\Sigma"), ("Phi", "Φ", "\\Phi"),
        ("Psi", "Ψ", "\\Psi"), ("Omega", "Ω", "\\Omega"),
    ]
    for name, char, latex in greek:
        syms.append(Symbol(name=name, char=char, latex=latex))

    # --- Common math operators & relations --------------------------------
    math = [
        ("times", "×", "\\times"), ("div", "÷", "\\div"),
        ("cdot", "⋅", "\\cdot"), ("pm", "±", "\\pm"),
        ("mp", "∓", "\\mp"), ("leq", "≤", "\\leq"),
        ("geq", "≥", "\\geq"), ("neq", "≠", "\\neq"),
        ("approx", "≈", "\\approx"), ("equiv", "≡", "\\equiv"),
        ("propto", "∝", "\\propto"), ("infty", "∞", "\\infty"),
        ("partial", "∂", "\\partial"), ("nabla", "∇", "\\nabla"),
        ("sum", "∑", "\\sum"), ("prod", "∏", "\\prod"),
        ("int", "∫", "\\int"), ("sqrt", "√", "\\sqrt"),
        ("in", "∈", "\\in"), ("notin", "∉", "\\notin"),
        ("subset", "⊂", "\\subset"), ("supset", "⊃", "\\supset"),
        ("cup", "∪", "\\cup"), ("cap", "∩", "\\cap"),
        ("forall", "∀", "\\forall"), ("exists", "∃", "\\exists"),
        ("rightarrow", "→", "\\rightarrow"),
        ("leftarrow", "←", "\\leftarrow"),
        ("leftrightarrow", "↔", "\\leftrightarrow"),
        ("Rightarrow", "⇒", "\\Rightarrow"),
        ("mapsto", "↦", "\\mapsto"),
    ]
    for name, char, latex in math:
        syms.append(Symbol(name=name, char=char, latex=latex))

    return syms


_TABLE: List[Symbol] = _build_table()
_BY_NAME: Dict[str, Symbol] = {s.name: s for s in _TABLE}
_BY_CHAR: Dict[str, Symbol] = {s.char: s for s in _TABLE}


def all_symbols() -> List[Symbol]:
    """Every symbol HandTex knows about (stable order)."""
    return list(_TABLE)


def by_name(name: str) -> Optional[Symbol]:
    return _BY_NAME.get(name)


def by_char(char: str) -> Optional[Symbol]:
    return _BY_CHAR.get(char)


def resolve(token: str) -> Optional[Symbol]:
    """Resolve a token that may be a name OR a literal character."""
    if token in _BY_NAME:
        return _BY_NAME[token]
    if token in _BY_CHAR:
        return _BY_CHAR[token]
    return None


def latex_for(name: str) -> str:
    """LaTeX command/literal for a symbol name (falls back to the name)."""
    s = _BY_NAME.get(name)
    return s.latex if s else name


def names() -> Iterable[str]:
    return _BY_NAME.keys()
