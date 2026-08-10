import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINER_STATE = r"D:\Dev\sllm_training_v2\teamspace_uploads_Qwen3.5-4B_v16v2_clean_dataset_220726_3epochs\trainer_state.json"
OUT_DIR = r"D:\Dev\sllm_training_v2\report_assets"

with open(TRAINER_STATE, "r", encoding="utf-8") as f:
    state = json.load(f)

log_history = state["log_history"]
steps_log = [e for e in log_history if "loss" in e]

steps = [e["step"] for e in steps_log]
losses = [e["loss"] for e in steps_log]
lrs = [e["learning_rate"] for e in steps_log]
epochs = [e["epoch"] for e in steps_log]

# palette (same as chart_gen.py)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


# ---- 1. training loss curve (log scale) with epoch boundaries ----
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(steps, losses, color=BLUE, linewidth=1.8, zorder=3)
ax.set_yscale("log")
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0, which="both")
ax.set_axisbelow(True)
style_ax(ax)
for ep_boundary in [128, 256]:
    ax.axvline(ep_boundary, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
ax.text(128, max(losses) * 0.75, " epoch 2 →", fontsize=8.5, color=INK_MUTED)
ax.text(256, max(losses) * 0.75, " epoch 3 →", fontsize=8.5, color=INK_MUTED)
ax.set_title("Training loss (log scale) — 384 steps, 3 epochs", fontsize=13, color=INK,
             loc="left", fontweight="bold", pad=12)
ax.set_xlabel("Training step", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("Loss (log scale)", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_loss_curve.png"), dpi=160)
plt.close(fig)

# ---- 2. learning rate schedule ----
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(steps, lrs, color=ORANGE, linewidth=1.8, zorder=3)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_title("Learning-rate schedule — warmup + cosine decay", fontsize=12, color=INK,
             loc="left", fontweight="bold", pad=10)
ax.set_xlabel("Training step", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("Learning rate", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_lr_schedule.png"), dpi=160)
plt.close(fig)

# ---- per-epoch average loss (excluding the first 3 warmup-affected log points) ----
import collections
epoch_bucket = collections.defaultdict(list)
for e, l in zip(epochs, losses):
    epoch_bucket[int(e) + 1 if e == int(e) else int(e) + 1].append(l)

# simpler: bucket by ceil(epoch)
buckets = {1: [], 2: [], 3: []}
for e, l in zip(epochs, losses):
    ep_num = min(3, int(e) + 1)
    buckets[ep_num].append(l)

summary = {}
for ep, vals in buckets.items():
    if vals:
        summary[ep] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}

print(json.dumps(summary, indent=2))
print("total logged points:", len(losses))
print("final 10 losses:", losses[-10:])
print("train_runtime_hours:", 52860.0556 / 3600)
