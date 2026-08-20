"""Dataflow (cone-of-influence) analysis over a live slang elaboration.

Builds one signal-level dataflow graph for a whole elaborated design and answers
reachability questions against it:

    * which top-level inputs can affect a given submodule (backward cone)
    * which top-level outputs it can affect (forward cone)
    * which module instances lie entirely outside both (the blackboxable set)

This is the analysis half of scope-reduced formal verification: a cut made
inside submodule M can only change outputs in M's forward cone, so every
instance outside that cone is structurally identical between the golden and cut
designs and can be abstracted away during equivalence checking.

The graph is built from the *hierarchical* elaborated AST (via
``elaborator.build_compilation``), not from the flattened re-emitted source, so
module boundaries survive and queries can be posed against any instance at any
depth.

CLI::

    python -m papercuts.scope design.sv --list
    python -m papercuts.scope design.sv --module top.u_mul
    python -m papercuts.scope design.sv --module top.u_mul --dump

Precision and soundness: every construct this module does not recognize falls
back to a conservative all-refs-to-all-refs connection, which can only make a
cone *larger*. That direction is safe -- an over-large cone blackboxes fewer
modules and loses performance, while an under-large cone would blackbox logic
that actually carries the cut's effect and silently produce a false "proven".
Run with ``--dump`` to see which kinds hit the fallback.

The same asymmetry governs ``--ignore-diag``. Slang refuses to hand back an
elaboration it considers erroneous, and real RTL trips errors that have no
bearing on dataflow -- ``MissingTimeScale`` is pure timing metadata, for
instance. Suppressing those is safe. Suppressing an error that means part of the
design failed to elaborate is not: the missing logic yields a cone that is too
*small*, which is the direction that produces false "proven" verdicts. In
particular, do not suppress ``UnknownModule`` to push past an absent definition
-- an instance whose body never elaborated contributes no edges at all. Supply
the missing source, or name the module with ``--exclude-module`` so it routes
through the conservative opaque-instance path instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pyslang

from papercuts.elaborator import (
    ElaborationError,
    build_compilation,
)

#: Member kinds that carry no dataflow of their own: declarations, imports and
#: type machinery. Any driver they have appears as a separate ContinuousAssign
#: or ProceduralBlock member, or as an initializer handled explicitly. Listed
#: exhaustively so that genuinely unrecognized kinds stay visible in --dump
#: instead of being dropped in silence. Deliberately excluded, because they do
#: move data and should surface as unhandled until handled: NetAlias, Modport,
#: CheckerInstance.
INERT_MEMBER_KINDS = {
    "Attribute", "AssertionPort", "ClockingBlock", "ClockVar",
    "CompilationUnit", "ConstraintBlock", "DefParam", "Definition",
    "ElabSystemTask", "EmptyMember", "EnumValue", "ExplicitImport",
    "ForwardingTypedef", "GenericClassDef", "Genvar", "InstanceBody",
    "InterfacePort", "LetDecl", "MethodPrototype", "MultiPort", "Package",
    "Parameter", "Port", "Primitive", "PrimitivePort", "Property", "Sequence",
    "SpecifyBlock", "Specparam", "StatementBlock", "Subroutine", "TypeAlias",
    "TypeParameter", "TransparentMember", "WildcardImport",
}


def _conn_expr(conn):
    """Normalize one port connection down to an ``Expression``.

    The shape depends on which symbol owns the connection
    (``slang/ast/symbols/InstanceSymbols.h``)::

        InstanceSymbol          :118  PortConnection -- expression in .expression
        UninstantiatedDefSymbol :283  AssertionExpr
        PrimitiveInstanceSymbol :338  bare Expression

    The middle case is the surprising one: slang cannot know that an unknown
    definition is not a checker, so it binds the connections as assertion
    expressions and the real ``Expression`` sits inside ``SimpleAssertionExpr``
    (``ASTBindings.cpp:400``). ``AssertionExpr`` has no ``.visit``, so failing
    to unwrap it either raises or yields no refs at all.
    """
    ex = getattr(conn, "expression", conn)
    if ex is not None and kind(ex) == "Simple":
        ex = getattr(ex, "expr", None)
    return ex


#: Symbol kinds that carry a runtime value and therefore become graph nodes.
#: Parameters, genvars and specparams are resolved constants in an elaborated
#: design (each instance is specialized to one value) and carry no dataflow.
VALUE_KINDS = {
    "Net",
    "Variable",
    "Port",
    "MultiPort",
    "FormalArgument",
    "ModportPort",
}

#: Expression kinds that are a direct reference to a value symbol.
REF_KINDS = {"NamedValue", "HierarchicalValue"}

#: Port directions that read from the enclosing scope / drive into it.
IN_DIRS = {"In", "InOut", "Ref"}
OUT_DIRS = {"Out", "InOut", "Ref"}


def kind(node) -> str:
    """Enum name of a symbol/expression/statement kind, e.g. 'BinaryOp'."""
    return node.kind.name


def canon(sym) -> str:
    """Canonical graph key for a value symbol.

    A module port and the net/variable it drives are two *distinct* symbols in
    slang: references inside the module body resolve to the internal net, while
    port connections attach to the ``PortSymbol``. Collapsing a port onto its
    internal symbol makes them a single graph node, so a signal crossing a
    module boundary is not split into two disconnected halves.

    ``hierarchicalPath`` is the key rather than the symbol object itself because
    pyslang symbol identity is not stable across accesses, while the path is a
    stable unique name for the symbol.
    """
    internal = getattr(sym, "internalSymbol", None)
    if internal is not None:
        sym = internal
    return sym.hierarchicalPath


# MARK: Port record
@dataclass(frozen=True)
class Port:
    """One port of one instance, as a graph node plus its display name."""

    node: str        # canonical graph key (the internal net's hierarchical path)
    direction: str   # "In" | "Out" | "InOut" | "Ref"
    name: str        # declared port name
    path: str        # the port symbol's own hierarchical path

    @property
    def is_input(self) -> bool:
        return self.direction in IN_DIRS

    @property
    def is_output(self) -> bool:
        return self.direction in OUT_DIRS


# MARK: Cone result
@dataclass
class Cone:
    """Result of a cone query against one target instance."""

    target: str                     # instance path the query was posed for
    affecting_inputs: list[Port]    # design inputs in the target's backward cone
    affected_outputs: list[Port]    # design outputs in the target's forward cone
    blackboxable: list[str]         # instance paths safe to abstract away
    forward: set[str] = field(default_factory=set, repr=False)
    backward: set[str] = field(default_factory=set, repr=False)

    def summary(self) -> str:
        return (
            f"{self.target}: {len(self.affecting_inputs)} affecting input(s), "
            f"{len(self.affected_outputs)} affected output(s), "
            f"{len(self.blackboxable)} blackboxable instance(s)"
        )


# MARK: Graph
class ScopeGraph:
    """Signal-level dataflow graph for one elaborated design.

    Nodes are canonical signal paths (see :func:`canon`); an edge ``a -> b``
    means "a can affect b". Because every node is keyed by its full hierarchical
    path, cross-module references -- port connections and hierarchical
    references alike -- land on the same nodes with no special handling.
    """

    def __init__(self) -> None:
        self.fanout: dict[str, set[str]] = defaultdict(set)
        self.fanin: dict[str, set[str]] = defaultdict(set)
        #: instance path -> its ports
        self.ports: dict[str, list[Port]] = {}
        #: instance path -> module definition name
        self.definition: dict[str, str] = {}
        #: top-level instance paths (the design boundary)
        self.tops: list[str] = []
        #: instances with no walkable body (blackboxes, excluded modules)
        self.opaque: set[str] = set()
        #: unhandled AST kinds that hit the conservative fallback, with counts
        self.unhandled: dict[str, int] = defaultdict(int)
        #: owning Compilation, set by build_graph. pyslang symbols are
        #: non-owning references into its arena, so it must outlive the graph.
        self._comp = None

    # --- construction --------------------------------------------------------

    @classmethod
    def build(cls, comp, opaque_defs=(), tops=()) -> "ScopeGraph":
        """Build the graph for every top instance in ``comp``.

        ``opaque_defs`` names module *definitions* whose bodies must not be
        walked (vendor IP, ``--exclude-module`` targets). Each such instance is
        wired input-to-output all-to-all, matching how the emitter treats a
        module it cannot see inside.

        ``tops`` restricts the roots by module name. Slang makes every module
        that nothing instantiates a top, so a source directory holding unused
        or leftover modules yields several roots -- and then their ports count
        as design I/O and their instances appear as blackboxable, which is
        meaningless for a proof of one specific top. Empty means "all roots".
        """
        g = cls()
        opaque = set(opaque_defs)
        roots = list(comp.getRoot().topInstances)
        wanted = set(tops)
        if wanted:
            roots = [i for i in roots if i.name in wanted]
            missing = wanted - {i.name for i in roots}
            if missing:
                found = ", ".join(sorted(i.name for i in comp.getRoot().topInstances))
                raise ElaborationError(
                    f"top module(s) {sorted(missing)} are not roots of this "
                    f"compilation. Roots found: {found or '(none)'}. A module "
                    f"that something else instantiates is not a root."
                )
        for inst in roots:
            g.tops.append(inst.hierarchicalPath)
            g._add_instance(inst, opaque)
        return g

    def add_edge(self, src: str, dst: str) -> None:
        if src == dst:
            return
        self.fanout[src].add(dst)
        self.fanin[dst].add(src)

    def add_edges(self, srcs, dsts) -> None:
        for d in dsts:
            for s in srcs:
                self.add_edge(s, d)

    def _add_instance(self, inst, opaque_defs: set) -> None:
        path = inst.hierarchicalPath
        body = inst.body
        self.definition[path] = body.name
        ports = self._record_ports(inst, body)

        if body.name in opaque_defs:
            # No visible body: assume every input reaches every output.
            self.opaque.add(path)
            self.add_edges(
                [p.node for p in ports if p.is_input],
                [p.node for p in ports if p.is_output],
            )
            return

        self._walk_members(body, opaque_defs)

    def _record_ports(self, inst, body) -> list[Port]:
        ports: list[Port] = []
        for m in body:
            if kind(m) not in ("Port", "MultiPort"):
                continue
            ports.append(
                Port(
                    node=canon(m),
                    direction=m.direction.name,
                    name=m.name,
                    path=m.hierarchicalPath,
                )
            )
        self.ports[inst.hierarchicalPath] = ports
        return ports

    def _walk_members(self, scope, opaque_defs: set) -> None:
        """Walk one scope's members, adding edges. Nested instances recurse."""
        for m in scope:
            k = kind(m)
            if k == "ContinuousAssign":
                self._assignment(m.assignment, set())
            elif k == "ProceduralBlock":
                self._statement(m.body, set())
            elif k == "Instance":
                self._connect(m)
                self._add_instance(m, opaque_defs)
            elif k in ("UninstantiatedDef", "PrimitiveInstance"):
                # No body to walk: an unknown definition, or a gate primitive.
                self._connect_opaque(m)
            elif k in ("Net", "Variable"):
                # `wire w = a & b;` keeps its driver as the symbol's own
                # initializer instead of emitting a separate ContinuousAssign
                # member -- which is why elaborator._emit_member reads the
                # initializer off the net too. Missing this drops real edges.
                init = getattr(m, "initializer", None)
                if init is not None:
                    reads: set[str] = set()
                    self._refs(init, reads)
                    self.add_edges(reads, {canon(m)})
            elif k in ("GenerateBlock", "GenerateBlockArray", "InstanceArray"):
                # Each is a Scope whose members are ordinary module members
                # (for InstanceArray, the per-element Instances of `u[3:0]`).
                # Uninstantiated generate branches carry no elaborated logic.
                if getattr(m, "isUninstantiated", False):
                    continue
                self._walk_members(m, opaque_defs)
            elif k not in INERT_MEMBER_KINDS:
                # Anything unrecognized is recorded rather than dropped. A
                # missed member is a *missing* edge, which shrinks cones -- the
                # direction that produces false "proven" verdicts -- so it must
                # stay visible in --dump rather than failing silently.
                self.unhandled[f"member:{k}"] += 1

    def _connect(self, inst) -> None:
        """Edges across an instance boundary, from its port connections."""
        for conn in inst.portConnections:
            ex = conn.expression
            if ex is None:
                continue
            port = conn.port
            node = canon(port)
            direction = getattr(port, "direction", None)
            direction = direction.name if direction is not None else "InOut"

            if direction in IN_DIRS:
                # Parent expression drives the child's port.
                reads: set[str] = set()
                self._refs(ex, reads)
                self.add_edges(reads, [node])

            if direction in OUT_DIRS:
                # An output connection serializes as an Assignment of the
                # connected lvalue to an EmptyArgument placeholder; the lvalue
                # is what the child's port drives.
                target = ex.left if kind(ex) == "Assignment" else ex
                if kind(target) == "EmptyArgument":
                    continue
                writes: set[str] = set()
                reads = set()
                self._lvalue(target, writes, reads)
                self.add_edges(set(reads) | {node}, writes)

    def _connect_opaque(self, u) -> None:
        """An instantiation with no available definition: fully connected.

        Its ports were never elaborated, so direction is unknown -- every
        referenced signal is treated as both driving and driven.
        """
        self.opaque.add(u.hierarchicalPath)
        refs: set[str] = set()
        for conn in getattr(u, "portConnections", ()) or ():
            ex = _conn_expr(conn)
            if ex is not None:
                self._refs(ex, refs)
        self.add_edges(refs, refs)

    # --- expressions ---------------------------------------------------------

    def _refs(self, node, out: set) -> None:
        """Collect every value symbol referenced anywhere under ``node``.

        Used for read sets. Visiting descends through sub-expressions,
        statements and nested symbols alike.
        """

        def cb(n):
            if kind(n) in REF_KINDS:
                sym = n.symbol
                if sym is not None and kind(sym) in VALUE_KINDS:
                    out.add(canon(sym))

        node.visit(cb)

    def _lvalue(self, expr, writes: set, reads: set) -> None:
        """Split an assignment target into the signals it *writes* and the
        signals it merely *reads*.

        The distinction matters: in ``mem[addr] <= d`` the target is ``mem``,
        but ``addr`` is read, not written. Collecting every reference under the
        lvalue would wrongly mark ``addr`` as driven and invent backwards edges.
        """
        k = kind(expr)
        if k in REF_KINDS:
            sym = expr.symbol
            if sym is not None and kind(sym) in VALUE_KINDS:
                writes.add(canon(sym))
        elif k == "ElementSelect":
            self._lvalue(expr.value, writes, reads)
            self._refs(expr.selector, reads)
        elif k == "RangeSelect":
            self._lvalue(expr.value, writes, reads)
            self._refs(expr.left, reads)
            self._refs(expr.right, reads)
        elif k == "MemberAccess":
            self._lvalue(expr.value, writes, reads)
        elif k == "Concatenation":
            for op in expr.operands:
                self._lvalue(op, writes, reads)
        elif k == "Replication":
            self._lvalue(expr.concat, writes, reads)
            self._refs(expr.count, reads)
        elif k == "StreamingConcatenation":
            for s in expr.streams:
                self._lvalue(s.operand, writes, reads)
        elif k == "Conversion":
            self._lvalue(expr.operand, writes, reads)
        elif k == "Assignment":
            self._assignment(expr, set())
        elif k == "EmptyArgument":
            return
        else:
            # Unrecognized lvalue form: assume everything under it is driven.
            self.unhandled[f"lvalue:{k}"] += 1
            self._refs(expr, writes)

    def _assignment(self, expr, control: set) -> None:
        """Add edges for one assignment, under an enclosing control set."""
        writes: set[str] = set()
        reads: set[str] = set(control)
        self._lvalue(expr.left, writes, reads)
        self._refs(expr.right, reads)
        self.add_edges(reads, writes)

    def _expr_effect(self, expr, control: set) -> None:
        """Edges for an expression evaluated for its side effects."""
        k = kind(expr)
        if k == "Assignment":
            self._assignment(expr, control)
        else:
            # A call may write through output arguments, and anything else here
            # is unrecognized. Connect all-to-all rather than risk missing a
            # driver.
            self.unhandled[f"expr:{k}"] += 1
            self._fallback(expr, control)

    def _fallback(self, node, control: set) -> None:
        """Conservative catch-all: everything referenced may drive everything."""
        refs: set[str] = set()
        self._refs(node, refs)
        self.add_edges(refs | control, refs)

    # --- statements ----------------------------------------------------------

    def _statement(self, stmt, control: set) -> None:
        """Walk a statement tree, threading control dependence.

        ``control`` carries the signals every enclosing condition reads, so
        ``if (en) y <= a;`` records ``en -> y`` as well as ``a -> y``. That is
        what a formal cone of influence means: a signal that decides *whether*
        an assignment happens is in the cone of what it assigns.
        """
        if stmt is None:
            return
        k = kind(stmt)

        if k == "ExpressionStatement":
            self._expr_effect(stmt.expr, control)

        elif k == "Block":
            self._statement(stmt.body, control)

        elif k == "List":
            for s in stmt.list:
                self._statement(s, control)

        elif k == "Conditional":
            inner = set(control)
            for cond in stmt.conditions:
                self._refs(cond.expr, inner)
            self._statement(stmt.ifTrue, inner)
            self._statement(stmt.ifFalse, inner)

        elif k in ("Case", "PatternCase"):
            inner = set(control)
            self._refs(stmt.expr, inner)
            for item in stmt.items:
                item_ctl = set(inner)
                for e in getattr(item, "expressions", ()) or ():
                    self._refs(e, item_ctl)
                self._statement(item.stmt, item_ctl)
            self._statement(stmt.defaultCase, inner)

        elif k == "ForLoop":
            inner = set(control)
            for init in stmt.initializers:
                self._expr_effect(init, control)
            if stmt.stopExpr is not None:
                self._refs(stmt.stopExpr, inner)
            for step in stmt.steps:
                self._expr_effect(step, inner)
            self._statement(stmt.body, inner)

        elif k in ("WhileLoop", "DoWhileLoop"):
            inner = set(control)
            self._refs(stmt.cond, inner)
            self._statement(stmt.body, inner)

        elif k == "RepeatLoop":
            inner = set(control)
            self._refs(stmt.count, inner)
            self._statement(stmt.body, inner)

        elif k in ("ForeverLoop", "ForeachLoop"):
            self._statement(stmt.body, control)

        elif k == "Timed":
            # The clock/reset event is a control dependence on everything the
            # block assigns.
            inner = set(control)
            if stmt.timing is not None:
                self._refs(stmt.timing, inner)
            self._statement(stmt.stmt, inner)

        elif k in ("Empty", "VariableDeclaration", "Return", "Break", "Continue",
                   "Disable", "Timing"):
            return

        else:
            self.unhandled[f"stmt:{k}"] += 1
            self._fallback(stmt, control)

    # --- queries -------------------------------------------------------------

    def reachable(self, starts, forward: bool = True) -> set[str]:
        """Signals reachable from ``starts``, following edges in one direction.

        Plain BFS with a visited set: sequential feedback makes the graph
        cyclic, so there is no topological order to exploit.
        """
        adj = self.fanout if forward else self.fanin
        seen = set(starts)
        queue = deque(seen)
        while queue:
            for nxt in adj.get(queue.popleft(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def boundary_ports(self) -> list[Port]:
        """Ports of the top instances -- the design's primary I/O."""
        out: list[Port] = []
        for top in self.tops:
            out.extend(self.ports.get(top, ()))
        return out

    def cone(self, target: str) -> Cone:
        """Cone of influence for instance ``target`` (a hierarchical path)."""
        if target not in self.ports:
            raise KeyError(
                f"no instance {target!r} in design; "
                f"use --list to see available instance paths"
            )
        ports = self.ports[target]
        forward = self.reachable(
            [p.node for p in ports if p.is_output], forward=True
        )
        backward = self.reachable(
            [p.node for p in ports if p.is_input], forward=False
        )
        boundary = self.boundary_ports()
        return Cone(
            target=target,
            affecting_inputs=[
                p for p in boundary if p.is_input and p.node in backward
            ],
            affected_outputs=[
                p for p in boundary if p.is_output and p.node in forward
            ],
            blackboxable=self._blackboxable(target, forward),
            forward=forward,
            backward=backward,
        )

    def _blackboxable(self, target: str, forward: set[str]) -> list[str]:
        """Instances that can be abstracted away when checking a cut in ``target``.

        An instance qualifies only if no port of it lies in the target's forward
        cone -- i.e. the cut provably cannot reach it, so its logic is identical
        in the golden and cut designs and a shared free variable
        over-approximates it soundly.

        Ancestors of the target are excluded because blackboxing a parent would
        hide the cut itself, which makes both sides trivially equal and reports
        a false "proven". Descendants are excluded for the same reason.
        """
        out: list[str] = []
        for path, ports in self.ports.items():
            if path == target or path in self.tops:
                continue
            if target.startswith(path + ".") or path.startswith(target + "."):
                continue
            if any(p.node in forward for p in ports):
                continue
            out.append(path)
        return sorted(out)


# MARK: entry points
def resolve_diag_codes(names):
    """Map slang diagnostic names (``"MissingTimeScale"``) to ``DiagCode``s."""
    codes = set()
    for name in names:
        code = getattr(pyslang.Diags, name, None)
        if code is None:
            raise ElaborationError(
                f"unknown diagnostic name {name!r}. Names are slang's, as they "
                f"appear in its diagnostics table -- e.g. MissingTimeScale, "
                f"UsedBeforeDeclared, UnknownModule."
            )
        codes.add(code)
    return codes


def report_diagnostics(comp, ignored=()):
    """Print diagnostics; return True if any *non-ignored* error remains.

    This is ``elaborator.report_diagnostics`` plus a filter. It cannot be
    implemented by remapping severities on the engine: ``Diagnostic.isError()``
    is derived from the code's *default* severity in slang
    (``Diagnostics.cpp:38``) and ignores engine mappings entirely, so a
    suppressed code has to be dropped before the check rather than downgraded.
    """
    ignored = set(ignored)
    client = pyslang.TextDiagnosticClient()
    engine = pyslang.DiagnosticEngine(comp.sourceManager)
    engine.addClient(client)
    errored = False
    for d in comp.getAllDiagnostics():
        if d.code in ignored:
            continue
        engine.issue(d)
        errored = errored or d.isError()
    text = client.getString()
    if text.strip():
        print(text, file=sys.stderr)
    return errored


def build_graph(
    files, *, opaque_defs=(), ignore_diags=(), tops=(), incdirs=(), defines=()
) -> ScopeGraph:
    """Elaborate ``files`` and build the dataflow graph for the whole design.

    ``incdirs`` and ``defines`` mirror the elaborator's ``-I``/``-D``, for
    designs whose sources only parse with include paths or predefined macros.
    Neither is needed when the files come from ``outputs/concrete_sources``,
    which the elaborator has already preprocessed.
    """
    comp = build_compilation(list(files), incdirs=incdirs, defines=defines)
    if report_diagnostics(comp, resolve_diag_codes(ignore_diags)):
        raise ElaborationError("input has compilation errors; aborting")
    graph = ScopeGraph.build(comp, opaque_defs=opaque_defs, tops=tops)
    # Hold the compilation alive: pyslang symbols are non-owning references into
    # its arena, and the graph is derived from them.
    graph._comp = comp
    return graph


def blackboxable_definitions(graph: ScopeGraph, cone: Cone) -> list[str]:
    """Module *definition* names safe to hand to ``elaborate -bbox_m``.

    ``Cone.blackboxable`` names instances, but Jasper's ``-bbox_m`` takes a
    module name and abstracts away *every* instance of it. Those coincide
    for a design elaborated with ``flatten=True``, which emits one uniquely
    named definition per instance -- but not for modules the elaborator emitted
    verbatim (``--exclude-module`` and its subtree), which keep their original
    name and are shared by all their instances.

    So a name is returned only when every instance carrying it is blackboxable.
    Blackboxing a name with even one in-cone instance would abstract away logic
    the cut can reach; both sides of the equivalence check would then agree
    trivially and report a false "proven".
    """
    instances_by_def: dict[str, set[str]] = defaultdict(set)
    for path, def_name in graph.definition.items():
        instances_by_def[def_name].add(path)

    safe = set(cone.blackboxable)
    return sorted(
        name for name, paths in instances_by_def.items() if paths <= safe
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cone-of-influence analysis over a slang elaboration"
    )
    parser.add_argument("files", nargs="+", help="SystemVerilog source files")
    parser.add_argument(
        "--module", "-m", metavar="PATH",
        help="instance path to analyze (e.g. top.u_mul); see --list",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="list every instance path in the design and exit",
    )
    parser.add_argument(
        "--exclude-module", action="append", default=[], metavar="MODULE",
        help="module definition to treat as opaque (all inputs reach all "
             "outputs). Repeatable.",
    )
    parser.add_argument(
        "--top", action="append", default=[], metavar="MODULE",
        help="restrict the design to this top module. Repeatable. Without it "
             "every module nothing instantiates becomes a root, and unrelated "
             "roots' ports count as design I/O.",
    )
    parser.add_argument(
        "-I", "--incdir", action="append", default=[], metavar="DIR",
        help="add an include-search directory (like +incdir). Repeatable.",
    )
    parser.add_argument(
        "-D", "--define", action="append", default=[], metavar="NAME[=VALUE]",
        help="predefine a macro before preprocessing (like +define). Repeatable.",
    )
    parser.add_argument(
        "--ignore-diag", action="append", default=[], metavar="NAME",
        help="slang diagnostic to suppress, by name (e.g. MissingTimeScale). "
             "Repeatable. Only suppress diagnostics that cannot affect "
             "dataflow -- see the module docstring.",
    )
    parser.add_argument(
        "--dump", action="store_true",
        help="also print graph statistics and any unhandled AST kinds",
    )
    args = parser.parse_args()

    try:
        graph = build_graph(
            args.files,
            opaque_defs=args.exclude_module,
            ignore_diags=args.ignore_diag,
            tops=args.top,
            incdirs=args.incdir,
            defines=args.define,
        )
    except ElaborationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.list or not args.module:
        print(f"Top instance(s): {', '.join(graph.tops) or '(none)'}")
        print(f"{len(graph.ports)} instance(s):")
        for path in sorted(graph.ports):
            mark = " [opaque]" if path in graph.opaque else ""
            print(f"  {path}  ({graph.definition.get(path, '?')}){mark}")
        if not args.module:
            if not args.list:
                print("\nPass --module PATH to compute a cone.", file=sys.stderr)
            return 0

    try:
        cone = graph.cone(args.module)
    except KeyError as e:
        print(f"Error: {e.args[0]}", file=sys.stderr)
        return 1

    print(f"\n{cone.summary()}\n")
    print("Affecting inputs (design inputs in the backward cone):")
    for p in cone.affecting_inputs:
        print(f"  {p.name}")
    print("\nAffected outputs (design outputs in the forward cone):")
    for p in cone.affected_outputs:
        print(f"  {p.name}")
    print("\nBlackboxable instances (outside the forward cone):")
    for path in cone.blackboxable:
        print(f"  {path}  ({graph.definition.get(path, '?')})")

    if args.dump:
        edges = sum(len(v) for v in graph.fanout.values())
        print(f"\nGraph: {len(graph.fanout)} driver node(s), {edges} edge(s)")
        print(f"Forward cone: {len(cone.forward)} signal(s); "
              f"backward cone: {len(cone.backward)} signal(s)")
        if graph.opaque:
            print(f"Opaque instances: {', '.join(sorted(graph.opaque))}")
        if graph.unhandled:
            print("Unhandled kinds (conservative fallback applied):")
            for k, n in sorted(graph.unhandled.items(), key=lambda kv: -kv[1]):
                print(f"  {k}: {n}")
        else:
            print("Unhandled kinds: none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
