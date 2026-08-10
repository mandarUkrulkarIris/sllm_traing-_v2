"""
Shared color/style constants for visualize_comparison.py and visualize_table.py, so
both charts read as one visual system rather than drifting independently.
"""

# Fixed categorical color mapping - color follows the metric's identity everywhere in
# every chart, never reassigned, so "blue" always means "exact match" and nothing else.
COLOR_EXACT = "#2a78d6"
COLOR_LEXICAL = "#eb6834"
COLOR_EMBEDDING = "#1baf7a"

# Status colors - reserved for agreement banding only, never reused as a categorical
# series color. Always paired with a label (and, where practical, a distinct marker
# shape) rather than relying on hue alone.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Composite per-table agreement score thresholds (used by both the table-level
# distribution in visualize_comparison.py and the cell tinting in visualize_table.py).
EXCELLENT_THRESHOLD = 0.85
PARTIAL_THRESHOLD = 0.65
