"""Compare two papercuts runs of the same design, cut by cut.

Joins two ``papercuts.results.jsonl`` files on ``(module, idx)`` and reports
verdict agreement, wall-clock delta, and blackboxing coverage. The intended use
is a scope-reduction differential: one run with ``--scope-reduce`` and one
without, everything else identical.

    python -m papercuts.compare_runs outputs_full/papercuts.results.jsonl \\
                                     outputs_reduced/papercuts.results.jsonl

The line that matters is the soundness check. Blackboxing logic that the cut can
actually reach makes both sides of the equivalence check agree trivially, which
shows up as a cut that is "proven" under reduction but rejected at full scope.
That is the failure mode scope reduction has to be trusted not to produce; the
reverse (rejected under reduction, proven at full scope) only costs a missed
optimization and is already absorbed by the pipeline's automatic re-check.

Note on which verdict is compared: in a reduced run, a rejected cut is
automatically re-verified against the full design, and ``verdict`` holds that
*second* answer. The reduction-only answer is kept in ``verdict_reduced``, so
that is what this compares -- otherwise every rejection would trivially agree.
"""

import collections
import json
import sys


def load(path):
    """Per-cut records from a results.jsonl, keyed by (module, idx)."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("phase") == "cuts":
                out[(rec["module"], rec["idx"])] = rec
    return out


def reduced_verdict(rec):
    """The verdict the reduced run reached *before* any full-scope re-check."""
    return rec.get("verdict_reduced") or rec.get("verdict")


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    full = load(argv[1])
    red = load(argv[2])
    shared = sorted(set(full) & set(red))
    if not shared:
        print("No cuts in common -- are these runs of the same design?")
        return 1

    print(f"cuts in both:   {len(shared)}")
    print(f"full only:      {len(set(full) - set(red))}")
    print(f"reduced only:   {len(set(red) - set(full))}")

    matrix = collections.Counter()
    unsound = []
    rechecked = 0
    for key in shared:
        fv = full[key].get("verdict")
        rv = reduced_verdict(red[key])
        if red[key].get("verdict_reduced"):
            rechecked += 1
        matrix[(fv, rv)] += 1
        if rv == "proven" and fv != "proven":
            unsound.append((key, fv, rv))

    print(f"\n{'full':<14} {'reduced':<14} count")
    for (fv, rv), n in sorted(matrix.items(), key=lambda kv: -kv[1]):
        flag = ""
        if rv == "proven" and fv != "proven":
            flag = "   <-- SOUNDNESS BUG"
        print(f"{str(fv):<14} {str(rv):<14} {n:>5}{flag}")

    t_full = sum(full[k].get("elapsed") or 0 for k in shared)
    t_red = sum(red[k].get("elapsed") or 0 for k in shared)
    pct = (t_red / t_full - 1) * 100 if t_full else 0.0
    print(f"\nelapsed full:    {t_full:9.1f}s")
    print(f"elapsed reduced: {t_red:9.1f}s   ({t_red - t_full:+.1f}s, {pct:+.1f}%)")
    print(f"re-checks fired: {rechecked}")

    # A re-checked cut's reduced `elapsed` covers the reduced check *plus* a
    # full-scope re-run, so the raw total charges the re-check policy to
    # blackboxing. Estimate the reduced check alone by subtracting the same
    # cut's full-run time -- the closest available proxy for the re-run's cost.
    # Separates "is a blackboxed check faster?" from "does re-checking cost
    # more than reduction saves?", which are independently actionable.
    if rechecked:
        t_adj = 0.0
        for k in shared:
            e = red[k].get("elapsed") or 0
            if red[k].get("verdict_reduced"):
                e = max(0.0, e - (full[k].get("elapsed") or 0))
            t_adj += e
        adj_pct = (t_adj / t_full - 1) * 100 if t_full else 0.0
        print(
            f"  minus re-checks: {t_adj:9.1f}s   "
            f"({t_adj - t_full:+.1f}s, {adj_pct:+.1f}%)"
        )

    bbox = sorted(red[k].get("bbox") or 0 for k in shared)
    inst = next(
        (red[k].get("instances") for k in shared if red[k].get("instances")),
        "?",
    )
    print(
        f"blackboxed/cut:  min {bbox[0]}  "
        f"median {bbox[len(bbox) // 2]}  max {bbox[-1]}   "
        f"of {inst} instances"
    )

    print()
    if unsound:
        print("!! proven under reduction but NOT at full scope:")
        for (mod, idx), fv, rv in unsound:
            print(f"   {mod} idx={idx}   full={fv}  reduced={rv}")
        print("\nThe cone is letting a reachable module be blackboxed.")
        print("Do not trust the reduced verdicts until this is resolved.")
        return 1

    print("OK: nothing proved under reduction that failed at full scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
