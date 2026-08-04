"""Coverage for bit-shrink on parameterized packed ranges (`logic [WIDTH-1:0] x;`).

Syntax alone says neither how wide such a range is nor which way it runs, so the
collector only targets it when the caller supplies the evaluated bounds via
`symbolic_ranges` -- which the pipeline fills from a read-only elaboration. Both
halves matter: without a width, over-narrowing produces a reversed range like
`[-1:0]`, and without a direction, a range that is ALREADY reversed (the real-RTL
idiom of a placeholder `parameter WIDTH = 0`) gets narrowed at its low end. Either
way the result is legal SV that silently *widens* the signal instead of erroring.
"""

from papercuts import Papercutter
from papercuts.chipper import definition_signal_ranges
from papercuts.elaborator import build_compilation
from pyslang.syntax import SyntaxTree

SRC = """
module symprobe #(parameter WIDTH = 8) (
    input  logic [WIDTH-1:0] a,
    input  logic             s,
    output logic [WIDTH-1:0] y
    );
    logic [WIDTH-1:0]   sym;
    logic [7:0]         lit;
    logic               one;
    logic [WIDTH-1:0][3:0] multi;
    always_comb begin
        sym = a;
        lit = 8'h0;
        one = s;
        multi = '0;
        y = s ? sym : a;
    end
endmodule
"""


def _norm(s):
    return " ".join(s.split())


def _shrinks(pc):
    """{narrowed declaration text: cut index} for every bitshrink cut."""
    out = {}
    for i, (t, _) in enumerate(pc.cut_info()):
        if t != "bitshrink":
            continue
        text = _norm(pc.cut_index_text([i]))
        # The one declaration this cut rewrote is the one absent from the source.
        out[next(d for d in DECLS if d not in text)] = i
    return out


# Declarations as they appear in SRC; a bitshrink cut replaces exactly one.
DECLS = ["[WIDTH-1:0] sym", "[7:0] lit", "logic one", "[WIDTH-1:0][3:0] multi"]


def run():
    # Without widths, only literal ranges are shrinkable: `lit`, and the literal
    # SECOND dimension of `multi` (its parameterized first dimension is skipped).
    # `one` is a single bit, and `sym` has no width the collector can see.
    bare = _shrinks(Papercutter(SyntaxTree.fromText(SRC)))
    assert set(bare) == {"[7:0] lit", "[WIDTH-1:0][3:0] multi"}, sorted(bare)

    # With widths, `sym` becomes shrinkable too.
    pc = Papercutter(
        SyntaxTree.fromText(SRC),
        symbolic_ranges={"sym": (7, 0), "one": (0, 0), "multi": (31, 0),
                         "a": (7, 0), "y": (7, 0)},
    )
    cuts = _shrinks(pc)
    assert set(cuts) == {"[WIDTH-1:0] sym", "[7:0] lit", "[WIDTH-1:0][3:0] multi"}, sorted(cuts)

    sym_idx = cuts["[WIDTH-1:0] sym"]
    sym_text = _norm(pc.cut_index_text([sym_idx]))
    assert "[(WIDTH-1)-1:0] sym" in sym_text, sym_text
    # Untargeted declarations keep their original range verbatim.
    assert "[WIDTH-1:0] a" in sym_text and "[7:0] lit" in sym_text

    # `multi` is only ever narrowed on its literal dimension -- a per-signal width
    # cannot size one dimension of a multi-dimensional vector, so the symbolic one
    # is left alone even though a width was supplied.
    multi_text = _norm(pc.cut_index_text([cuts["[WIDTH-1:0][3:0] multi"]]))
    assert "[WIDTH-1:0][2:0] multi" in multi_text, multi_text

    # Multi-bit shrink subtracts the requested amount from the symbolic bound...
    assert "[(WIDTH-1)-3:0] sym" in _norm(pc.cut_index_text([sym_idx], {sym_idx: 3}))
    # ...and is clamped at width-1 so the range can never invert.
    widths = pc.cut_shrink_widths()
    assert widths[sym_idx] == 8, f"symbolic width should come from the map, got {widths[sym_idx]}"
    assert "[(WIDTH-1)-7:0] sym" in _norm(pc.cut_index_text([sym_idx], {sym_idx: 99}))

    # A width of 1 leaves nothing to remove, so the signal is not a candidate.
    narrow = _shrinks(Papercutter(SyntaxTree.fromText(SRC), symbolic_ranges={"sym": (0, 0)}))
    assert "[WIDTH-1:0] sym" not in narrow, "a 1-bit symbolic signal must not be shrinkable"

    # An ALREADY-reversed range (real RTL: `logic [W-1:0] x;` under a placeholder
    # `parameter W = 0`, giving [-1:0]) is 2 bits wide, but its symbolic LEFT is the
    # range's low end. Subtracting there would widen it to [-2:0], so it is skipped.
    rev = _shrinks(Papercutter(SyntaxTree.fromText(SRC), symbolic_ranges={"sym": (-1, 0)}))
    assert "[WIDTH-1:0] sym" not in rev, "a reversed range must not be shrinkable"

    # The mirror case is fine: for an ascending `[0:W-1]` the symbolic bound IS the
    # high end, so it narrows there.
    asc_src = SRC.replace("logic [WIDTH-1:0]   sym;", "logic [0:WIDTH-1]   sym;")
    asc = Papercutter(SyntaxTree.fromText(asc_src), symbolic_ranges={"sym": (0, 7)})
    asc_idx = next(
        i for i, (t, _) in enumerate(asc.cut_info())
        if t == "bitshrink" and "(WIDTH-1)" in asc.cut_index_text([i])
    )
    assert "[0:(WIDTH-1)-1] sym" in _norm(asc.cut_index_text([asc_idx]))

    # The legacy intermediate-wire strategy cannot narrow in place, so symbolic
    # widths are ignored there rather than producing a bogus `_papercuts` wire.
    legacy = Papercutter(
        SyntaxTree.fromText(SRC), shrink_with_intermediate=True, symbolic_ranges={"sym": (7, 0)}
    )
    assert all(
        "(WIDTH-1)" not in legacy.cut_index_text([i])
        for i, (t, _) in enumerate(legacy.cut_info())
        if t == "bitshrink"
    ), "intermediate-wire mode must not emit symbolic ranges"

    # End-to-end: the widths the pipeline actually passes come from elaboration,
    # and are the minimum across instances (a cut edits the one shared definition).
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sym.sv")
        with open(path, "w") as f:
            f.write(SRC + """
module symtop (input logic [7:0] a8, input logic [2:0] a3, input logic s,
               output logic [7:0] y8, output logic [2:0] y3);
    symprobe #(.WIDTH(8)) u8 (.a(a8), .s(s), .y(y8));
    symprobe #(.WIDTH(3)) u3 (.a(a3), .s(s), .y(y3));
endmodule
""")
        ranges = definition_signal_ranges(build_compilation([path]))
    assert ranges["symprobe"]["sym"] == (2, 0), (
        f"narrowest range across instances expected (2, 0), got {ranges['symprobe']['sym']}"
    )
    assert ranges["symprobe"]["lit"] == (7, 0), "a literal range is instance-independent"

    print("test_symbolic_bitshrink: OK")


if __name__ == "__main__":
    run()
