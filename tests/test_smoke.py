"""Smoke test for the analysis machinery itself.

Runs the script with `--synthetic` because that's the smallest, deterministic
input that exercises the full pivot → decompose → live-proxy → plot pipeline
without touching the real-data Parquet substrate. End-to-end validation on
real data is done by the cross-asset orchestrator, not here.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_synthetic_demo_runs():
    res = subprocess.run(
        [sys.executable, "scripts/overfitting.py", "--synthetic"],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert res.returncode == 0, res.stderr
    out = json.loads((ROOT / "decomposition.json").read_text())
    assert out["mode"] == "synthetic"
    parts = out["decomposition"]
    # Shares should sum to ~1 (allow some interaction slack)
    total = (parts["share_v_param"] + parts["share_v_strategy"]
             + parts["share_v_window"] + parts["share_v_finite"]
             + parts["share_residual"])
    assert 0.95 <= total <= 1.05, f"shares sum to {total}"
    assert parts["total"] > 0
    # Figures exist
    for fname in ["fig_decomposition_synthetic.png",
                  "fig_param_vs_live_synthetic.png",
                  "fig_decomp_by_family_synthetic.png"]:
        assert (ROOT / "figures" / fname).is_file(), f"missing {fname}"


if __name__ == "__main__":
    test_synthetic_demo_runs()
    print("OK")
