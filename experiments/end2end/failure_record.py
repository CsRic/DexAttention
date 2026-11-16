"""Appends the record of a training launch that ended without a timed step, so that the tables and figures can state why a
system has no result instead of leaving the reason to the reader.

The record's kind is OOM, TLE or unsupported when a line of the launch's log names that cause (out of memory; no plan
within the planner's time limit; fewer blocks than ranks) and that line is the reason; otherwise the kind is error and the
reason is the exit status with the log's last line. analysis/common.py (training_failures) reads the kind; the record has
no step_ms, so no timing or memory summary counts it.
Inputs : --log the launch's output   Outputs: one JSON line appended to --out
Example: python experiments/end2end/failure_record.py --workload w3 --system dcp --inputs a,b --rc 1 --log x.log --out launch1.jsonl
"""
import argparse
import json
import os

CAUSES = (("out of memory", "OOM"), ("did not finish within", "TLE"), ("fewer than", "unsupported"))


def reason_of(log, rc):
    """(kind, reason) of a failed launch from its log"""
    lines = []
    if os.path.exists(log):
        with open(log, errors="replace") as f:
            lines = [x.strip() for x in f if x.strip()]
    for x in lines:
        for text, kind in CAUSES:
            if text in x.lower():
                return kind, x[:300]
    return "error", (f"exit status {rc}: " + lines[-1][:260]) if lines else f"exit status {rc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--inputs", required=True, help="comma-separated input IDs of the launch")
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kind, reason = reason_of(a.log, a.rc)
    rec = dict(workload=a.workload.upper(), system=a.system, inputs=a.inputs.split(","), status="failed", rc=a.rc,
               kind=kind, reason=reason)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
