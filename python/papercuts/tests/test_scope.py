"""Cone-of-influence analysis (papercuts.scope).

Each design below has hand-computed cones. The properties that matter for
scope-reduced FV are asymmetric: a cone that is too *large* only costs
performance, but a cone that is too *small* would blackbox logic carrying a
cut's effect and silently report a false "proven". The assertions here check
both the exact expected sets and, separately, that nothing which must be in a
cone is missing.
"""

import os
import tempfile

from papercuts.scope import build_graph

# Two independent paths through the design: u_a feeds `y`, u_b feeds `z`.
# Nothing crosses between them, so each submodule's cone is exactly half.
SPLIT = """
module inv (input logic i, output logic o);
    assign o = ~i;
endmodule

module top (
    input  logic a,
    input  logic b,
    output logic y,
    output logic z
    );
    inv u_a (.i(a), .o(y));
    inv u_b (.i(b), .o(z));
endmodule
"""

# u_mid sits between u_src and u_sink, so its forward cone reaches `q` and its
# backward cone reaches `d`. Neither sibling is blackboxable from u_mid:
# u_sink is downstream, u_src is upstream but feeds it.
CHAIN = """
module stage (input logic i, output logic o);
    assign o = i;
endmodule

module top (input logic d, output logic q);
    logic w1, w2;
    stage u_src  (.i(d),  .o(w1));
    stage u_mid  (.i(w1), .o(w2));
    stage u_sink (.i(w2), .o(q));
endmodule
"""

# Control dependence and lvalue splitting. `en` gates the write to `y`, so it
# must appear in y's backward cone. `addr` indexes the lvalue in the memory
# write, so it is a read of addr -- not a write to it.
CONTROL = """
module top (
    input  logic       clk,
    input  logic       en,
    input  logic [3:0] a,
    input  logic [1:0] addr,
    input  logic [3:0] wdata,
    output logic [3:0] y,
    output logic [3:0] rdata
    );
    logic [3:0] mem [0:3];

    always_ff @(posedge clk) begin
        if (en)
            y <= a;
    end

    always_ff @(posedge clk) begin
        mem[addr] <= wdata;
    end

    assign rdata = mem[addr];
endmodule
"""


def _graph(src, name):
    """Build a ScopeGraph from inline source via a temp .sv file."""
    d = tempfile.mkdtemp(prefix="pc_scope_")
    path = os.path.join(d, f"{name}.sv")
    with open(path, "w") as f:
        f.write(src)
    return build_graph([path])


def _names(ports):
    return sorted(p.name for p in ports)


def test_split_cones():
    g = _graph(SPLIT, "split")
    cone = g.cone("top.u_a")

    assert _names(cone.affecting_inputs) == ["a"], _names(cone.affecting_inputs)
    assert _names(cone.affected_outputs) == ["y"], _names(cone.affected_outputs)
    # The other branch is untouched by a cut in u_a, so it is safe to abstract.
    assert cone.blackboxable == ["top.u_b"], cone.blackboxable


def test_chain_cones():
    g = _graph(CHAIN, "chain")
    cone = g.cone("top.u_mid")

    assert _names(cone.affecting_inputs) == ["d"], _names(cone.affecting_inputs)
    assert _names(cone.affected_outputs) == ["q"], _names(cone.affected_outputs)
    # u_sink is downstream of u_mid, so blackboxing it would hide the cut's
    # effect. u_src is upstream but feeds the target's forward cone start.
    assert "top.u_sink" not in cone.blackboxable, cone.blackboxable


def test_control_dependence_and_lvalue_split():
    g = _graph(CONTROL, "control")
    top = "top"
    cone = g.cone(top)

    ins = _names(cone.affecting_inputs)
    # Every primary input reaches something, so all appear in the top's own cone.
    assert "en" in ins and "a" in ins, ins

    # `en` gates the assignment to y: it must be in y's backward cone.
    y = next(p for p in g.ports[top] if p.name == "y")
    y_cone = g.reachable([y.node], forward=False)
    en = next(p for p in g.ports[top] if p.name == "en")
    a = next(p for p in g.ports[top] if p.name == "a")
    assert en.node in y_cone, "control dependence on `en` was not recorded"
    assert a.node in y_cone, "data dependence on `a` was not recorded"

    # `addr` indexes the lvalue `mem[addr]`: it is read there, so it belongs in
    # rdata's cone -- but the write target is `mem`, so `wdata` must not leak
    # into addr's own fanin.
    addr = next(p for p in g.ports[top] if p.name == "addr")
    wdata = next(p for p in g.ports[top] if p.name == "wdata")
    rdata = next(p for p in g.ports[top] if p.name == "rdata")
    rdata_cone = g.reachable([rdata.node], forward=False)
    assert addr.node in rdata_cone, "index read of `addr` was not recorded"
    assert wdata.node in rdata_cone, "`wdata` should reach `rdata` through mem"
    assert wdata.node not in g.fanin.get(addr.node, set()), (
        "`addr` was treated as written by `mem[addr] <= wdata`"
    )


def test_no_unhandled_kinds():
    """The conservative fallback should not fire on ordinary RTL."""
    for src, name in ((SPLIT, "split"), (CHAIN, "chain"), (CONTROL, "control")):
        g = _graph(src, name)
        assert not g.unhandled, f"{name}: unhandled kinds {dict(g.unhandled)}"


def run():
    test_split_cones()
    test_chain_cones()
    test_control_dependence_and_lvalue_split()
    test_no_unhandled_kinds()
    print("scope: OK")


if __name__ == "__main__":
    run()
