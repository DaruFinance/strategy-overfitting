"""Generate the per-asset D1-D10 lift bar chart (horizontal)."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
csv = ROOT / "per_asset_decile_data.csv"
df = pd.read_csv(csv).sort_values("D1_minus_D10_pp", ascending=True)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#cc4c4c" if x > 0 else "#999999" for x in df["D1_minus_D10_pp"]]
ax.barh(df["asset"], df["D1_minus_D10_pp"], color=colors, alpha=0.9)
for i, (a, v) in enumerate(zip(df["asset"], df["D1_minus_D10_pp"])):
    ax.text(v + (0.15 if v >= 0 else -0.15), i, f"{v:+.1f} pp",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color="black", linewidth=0.6)
ax.set_xlabel("D1 − D10 lift (percentage points)")
ax.set_title("V_param decile lift on live-proxy profitability\n"
             "(D1 = lowest in-sample sensitivity, predicting highest OOS profit)")
ax.grid(True, axis="x", alpha=0.3)
fig.tight_layout()
out = ROOT / "figures" / "fig_per_asset_lift.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
