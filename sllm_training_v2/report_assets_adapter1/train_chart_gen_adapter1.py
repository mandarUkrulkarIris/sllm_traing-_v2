import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = r"D:\Dev\sllm_training_v2_gitrepo\sllm_traing-_v2\sllm_training_v2"
TRAINER_STATE = os.path.join(BASE_DIR, "adapters", "Qwen3.5-4B_v16v2_clean_dataset_210826_v2", "trainer_state.json")
OUT_DIR = os.path.join(BASE_DIR, "report_assets_adapter1")

with open(TRAINER_STATE, "r", encoding="utf-8") as f:
    state = json.load(f)

log_history = state["log_history"]
train_log = [e for e in log_history if "loss" in e]
eval_log = [e for e in log_history if "eval_loss" in e]

train_steps = [e["step"] for e in train_log]
train_losses = [e["loss"] for e in train_log]
lrs = [e["learning_rate"] for e in train_log]
train_epochs = [e["epoch"] for e in train_log]

eval_steps = [e["step"] for e in eval_log]
eval_losses = [e["eval_loss"] for e in eval_log]
eval_acc = [e["eval_token_accuracy"] for e in eval_log]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3f8f5f"

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


TOTAL_STEPS = state.get("global_step", train_steps[-1])
EPOCH_STEPS = TOTAL_STEPS / 3.0

# ---- 1. train loss (log scale) with eval loss overlay ----
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(train_steps, train_losses, color=BLUE, linewidth=1.4, alpha=0.75, zorder=3, label="train loss (logged step)")
ax.plot(eval_steps, eval_losses, color=ORANGE, linewidth=2.2, marker="o", markersize=4, zorder=4,
        label="eval loss (306 held-out samples)")
ax.set_yscale("log")
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0, which="both")
ax.set_axisbelow(True)
style_ax(ax)
for k in (1, 2):
    ax.axvline(EPOCH_STEPS * k, color=BASELINE, linewidth=1, linestyle="--", zorder=2)
    ax.text(EPOCH_STEPS * k, max(train_losses) * 0.8, f" epoch {k+1} →", fontsize=8.5, color=INK_MUTED)
ax.set_title(f"Training vs. held-out eval loss (log scale) — {TOTAL_STEPS} steps, 3 epochs", fontsize=13,
             color=INK, loc="left", fontweight="bold", pad=12)
ax.set_xlabel("Training step", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("Loss (log scale)", fontsize=10, color=INK_SECONDARY)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_loss_curve_adapter1.png"), dpi=160)
plt.close(fig)

# ---- 2. learning rate schedule ----
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(train_steps, lrs, color=ORANGE, linewidth=1.8, zorder=3)
ax.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax.set_axisbelow(True)
style_ax(ax)
ax.set_title("Learning-rate schedule — warmup + cosine decay", fontsize=12, color=INK,
             loc="left", fontweight="bold", pad=10)
ax.set_xlabel("Training step", fontsize=10, color=INK_SECONDARY)
ax.set_ylabel("Learning rate", fontsize=10, color=INK_SECONDARY)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "training_lr_schedule_adapter1.png"), dpi=160)
plt.close(fig)

# ---- 3. eval loss + eval token accuracy across checkpoints ----
fig, ax1 = plt.subplots(figsize=(10, 4.2))
ax1.plot(eval_steps, eval_losses, color=ORANGE, linewidth=2, marker="o", markersize=4, zorder=3)
ax1.yaxis.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
ax1.set_axisbelow(True)
style_ax(ax1)
ax1.set_xlabel("Training step", fontsize=10, color=INK_SECONDARY)
ax1.set_ylabel("Eval loss", fontsize=10, color=ORANGE)
ax1.tick_params(axis="y", colors=ORANGE)

ax2 = ax1.twinx()
ax2.plot(eval_steps, [a * 100 for a in eval_acc], color=GREEN, linewidth=2, marker="s", markersize=4, zorder=3)
ax2.set_ylabel("Eval token accuracy (%)", fontsize=10, color=GREEN)
ax2.tick_params(axis="y", colors=GREEN)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color(BASELINE)
ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_visible(False)
ax2.tick_params(length=0)

ax1.set_title("Held-out eval loss and token accuracy across checkpoints (306 samples)", fontsize=12.5,
              color=INK, loc="left", fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "eval_loss_accuracy_adapter1.png"), dpi=160)
plt.close(fig)

# ---- per-epoch average train loss ----
buckets = {1: [], 2: [], 3: []}
for e, l in zip(train_epochs, train_losses):
    ep_num = min(3, int(e) + 1)
    buckets[ep_num].append(l)

summary = {}
for ep, vals in buckets.items():
    if vals:
        summary[ep] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}

print(json.dumps(summary, indent=2))
print("total logged train points:", len(train_losses))
print("eval checkpoints:", len(eval_losses))
print("eval_loss sequence:", eval_losses)
print("eval_token_accuracy sequence:", eval_acc)
train_runtime = next((e["train_runtime"] for e in log_history if "train_runtime" in e), None)
if train_runtime:
    print("train_runtime_hours:", train_runtime / 3600)
