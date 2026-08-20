"""`$unit`-scope declarations shared across several input files.

slang builds one `CompilationUnitSymbol` per `addSyntaxTree` -- i.e. per input
file -- so a header carrying `$unit`-scope decls that is `` `include ``d by N
files yields N identical copies of those decls. Two things follow, and both are
covered here:

1. The emitter concatenates every compilation unit's `$unit` members into one
   blob. The blob is a single file, so N copies are N redefinition errors. They
   must be merged by name, keeping first-occurrence order.

   This is why the whole file exists: the `compilationUnits` loop was added to
   fix `$unit` typedefs being *dropped*, and validated on a design whose decls
   happened to live in a single compilation unit -- so dropping was fixed and
   over-emitting introduced in the same change, invisibly. Every emitter test
   until now built its source from one `fromText`, i.e. one CU, which cannot
   reproduce it. Hence `_elaborate_multi`.

2. Merging must not be a silent first-wins. Two CUs can legitimately render the
   same name with *different* text -- different `` `define `` state at include
   time, or a differently-folded parameter. Dropping one would change the blob's
   semantics relative to the original design, so that case must raise.

Separately, `chipper.split_tree` replicates the `$unit` preamble into every
per-module file (each must parse standalone). A formal tool analyzes that whole
set into ONE global compilation unit -- Jasper's `analyze -y` does -- where the
copies collide exactly as in (1). The preamble is therefore wrapped in an
include guard, which collapses them there while staying a no-op for
single-file-compilation-unit tools like slang.

The golden assertion throughout is the one that was missing: re-parse the
emitted blob as a *single* file and require zero diagnostics.
"""

import contextlib
import io
import tempfile
from pathlib import Path

from papercuts import chipper
from papercuts.elaborator import EmitError, elaborate
from papercuts.utils import print_tree
from pyslang.ast import Compilation
from pyslang.syntax import SyntaxTree

# A header with `$unit`-scope decls (not in a package), `include`d by three
# files. Each including file gets its own CompilationUnitSymbol carrying a copy.
DEFS_VH = """
`ifndef DEFS_VH
`define DEFS_VH
localparam int UNIT_W = 8;
typedef struct packed { logic [UNIT_W-1:0] a; logic v; } unit_pkt_t;
`endif
"""

TOP_SV = """
`include "defs.vh"
module top (input unit_pkt_t i, output logic [UNIT_W-1:0] oa, ob);
    leaf_a ua (.i(i), .o(oa));
    leaf_b ub (.i(i), .o(ob));
endmodule
"""

LEAF_A_SV = """
`include "defs.vh"
module leaf_a (input unit_pkt_t i, output logic [UNIT_W-1:0] o);
    assign o = i.v ? i.a : '0;
endmodule
"""

LEAF_B_SV = """
`include "defs.vh"
module leaf_b (input unit_pkt_t i, output logic [UNIT_W-1:0] o);
    assign o = i.a ^ {UNIT_W{i.v}};
endmodule
"""

# Same decl name rendered differently per CU: b.sv defines WIDE before including,
# a.sv does not, so CW folds to 16 in one compilation unit and 8 in the other.
CONFLICT_DEFS_VH = """
`ifndef CDEFS_VH
`define CDEFS_VH
`ifdef WIDE
localparam int CW = 16;
`else
localparam int CW = 8;
`endif
`endif
"""

CONFLICT_A_SV = """
`include "cdefs.vh"
module ca (input [CW-1:0] x, output [CW-1:0] y);
    assign y = x;
endmodule
"""

CONFLICT_B_SV = """
`define WIDE
`include "cdefs.vh"
module cb (input [CW-1:0] x, output [CW-1:0] y);
    ca ua (.x(x[7:0]), .y(y[7:0]));
    assign y[CW-1:8] = '0;
endmodule
"""


def _elaborate_multi(files, top_first):
    """Elaborate a multi-FILE design, so the compilation has multiple CUs.

    ``files`` maps filename -> source. ``top_first`` is the order handed to the
    elaborator. Suppress input-diagnostic prints so the test stays quiet; real
    input errors still raise ElaborationError.
    """
    with tempfile.TemporaryDirectory() as d:
        for name, text in files.items():
            (Path(d) / name).write_text(text)
        paths = [str(Path(d) / n) for n in top_first]
        with contextlib.redirect_stderr(io.StringIO()):
            return elaborate(paths, incdirs=[d])


def _compile_errors(text):
    comp = Compilation()
    comp.addSyntaxTree(SyntaxTree.fromText(text))
    return [d for d in comp.getAllDiagnostics() if d.isError()]


def run():
    out = _elaborate_multi(
        {"defs.vh": DEFS_VH, "top.sv": TOP_SV,
         "leaf_a.sv": LEAF_A_SV, "leaf_b.sv": LEAF_B_SV},
        ["top.sv", "leaf_a.sv", "leaf_b.sv"],
    )

    # Three CUs each carry the decls; exactly one copy of each may be emitted.
    assert out.count("localparam int UNIT_W") == 1, \
        f"UNIT_W emitted {out.count('localparam int UNIT_W')}x, want 1:\n{out}"
    assert out.count("unit_pkt_t;") == 1, \
        f"unit_pkt_t emitted {out.count('unit_pkt_t;')}x, want 1:\n{out}"

    # Merged, not dropped: the modules still reference both, so both must survive.
    assert "unit_pkt_t i" in out, f"$unit typedef lost from ports:\n{out}"

    # Decls must precede the first module that uses them.
    assert out.index("unit_pkt_t;") < out.index("module "), \
        f"$unit decls emitted after the modules that use them:\n{out}"

    # Golden check: the blob must compile as ONE file. This is the invariant that
    # was missing -- during elaboration each CU is its own scope, so slang is
    # happy with the duplicates; only reading the blob back as a single file
    # catches them.
    errs = _compile_errors(out)
    assert not errs, f"emitted blob has {len(errs)} compile error(s) as one file"

    # --- conflict must raise, never silently first-win -----------------------
    try:
        _elaborate_multi(
            {"cdefs.vh": CONFLICT_DEFS_VH, "ca.sv": CONFLICT_A_SV,
             "cb.sv": CONFLICT_B_SV},
            ["cb.sv", "ca.sv"],
        )
    except EmitError as e:
        assert "CW" in str(e), f"conflict error does not name the symbol: {e}"
    else:
        raise AssertionError(
            "same name with different text across CUs was merged silently; "
            "that changes blob semantics vs the original design and must raise"
        )

    # --- split files: preamble replicated, but guarded -----------------------
    # Each per-module file needs the decls to parse standalone, yet a formal tool
    # analyzes the set into one global compilation unit, where the copies would
    # collide. The guard collapses them there.
    split = chipper.split_tree(SyntaxTree.fromText(out))
    assert len(split) >= 2, f"expected several modules from the blob, got {len(split)}"
    for name, tree in split:
        # print_tree is what the pipeline writes to disk, and the only view that
        # carries preprocessor directives -- str(tree.root) drops them as trivia.
        text = print_tree(tree)
        assert f"`ifndef {chipper.UNIT_SCOPE_GUARD}" in text, \
            f"split file {name} replicates the $unit preamble unguarded:\n{text}"
        assert "`endif" in text, f"split file {name} has an unterminated guard:\n{text}"
        assert "unit_pkt_t" in text, f"split file {name} lost the $unit decls:\n{text}"
        # Still valid on its own -- the guard is inert in a single-file parse, so
        # the decls are present exactly once. UnknownModule is expected and not a
        # defect here: a parent's children live in sibling files, which the formal
        # tool resolves through its `-y` library path, not from this file.
        errs = [d for d in _compile_errors(text) if "UnknownModule" not in str(d.code)]
        assert not errs, f"split file {name} has {len(errs)} non-instantiation error(s)"

    print("test_unit_scope: OK ($unit decls merged across CUs, conflict raises, "
          "split preamble guarded)")


if __name__ == "__main__":
    run()
