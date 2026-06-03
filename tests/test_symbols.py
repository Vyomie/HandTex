from handtex.font import symbols
from handtex.font.symbols import VHint


def test_symbols_have_unique_codepoints():
    seen = {}
    for s in symbols.all_symbols():
        if s.name == "space":
            continue
        assert s.codepoint not in seen, f"{s.name} clashes with {seen.get(s.codepoint)}"
        seen[s.codepoint] = s.name


def test_resolve_by_name_and_char():
    assert symbols.resolve("alpha").char == "α"
    assert symbols.resolve("α").name == "alpha"
    assert symbols.resolve("x").name == "x"


def test_latex_mapping():
    assert symbols.latex_for("alpha") == "\\alpha"
    assert symbols.latex_for("x") == "x"
    assert symbols.latex_for("sum") == "\\sum"


def test_vertical_hints():
    assert symbols.by_name("comma").vhint == VHint.LOW
    assert symbols.by_name("apostrophe").vhint == VHint.HIGH
    assert symbols.by_name("asterisk").vhint == VHint.HIGH
    assert symbols.by_name("x").vhint == VHint.NORMAL
