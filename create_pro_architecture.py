#!/usr/bin/env python3
"""
Generate professional CNN architecture diagram similar to academic paper style.
Creates a clean visualization with 3D feature map blocks and neural network nodes.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 8), dpi=150)
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'input': '#E3F2FD',
    'conv_shared': '#81C784',  # Green for shared conv
    'conv_a': '#FFAB91',       # Orange for task A
    'conv_b': '#CE93D8',       # Purple for task B
    'conv_c': '#FFF59D',       # Yellow for task C
    'dense': '#90CAF9',        # Blue for dense
    'output_a': '#EF5350',     # Red output
    'output_b': '#AB47BC',     # Purple output
    'output_c': '#FFC107',     # Yellow output
    'arrow': '#424242',
    'text': '#212121'
}

def draw_3d_block(ax, x, y, width, height, depth, color, label=None, sublabel=None):
    """Draw a 3D block representing feature maps."""
    # Number of layers to show depth
    n_layers = min(int(depth / 8) + 3, 8)
    offset = 0.08

    for i in range(n_layers):
        rect = FancyBboxPatch(
            (x + i * offset, y - i * offset),
            width, height,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=color,
            edgecolor='#2E7D32' if 'C784' in color else '#555555',
            linewidth=0.8,
            alpha=0.9 - i * 0.08
        )
        ax.add_patch(rect)

    # Add label above
    if label:
        ax.text(x + width/2 + n_layers*offset/2, y + height + 0.3, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors['text'])

    # Add sublabel below
    if sublabel:
        ax.text(x + width/2 + n_layers*offset/2, y - 0.25 - n_layers*offset, sublabel,
                ha='center', va='top', fontsize=8, color=colors['text'], style='italic')

def draw_arrow(ax, start, end, color='#424242', style='->', curved=False):
    """Draw an arrow between two points."""
    if curved:
        arrow = FancyArrowPatch(start, end,
                                arrowstyle='->', mutation_scale=15,
                                connectionstyle='arc3,rad=0.2',
                                color=color, linewidth=1.5)
    else:
        arrow = FancyArrowPatch(start, end,
                                arrowstyle='->', mutation_scale=15,
                                color=color, linewidth=1.5)
    ax.add_patch(arrow)

def draw_neural_layer(ax, x, y, n_nodes, height, color, label=None, node_labels=None):
    """Draw a vertical layer of neural network nodes."""
    spacing = height / (n_nodes + 1)
    nodes = []
    for i in range(n_nodes):
        node_y = y + spacing * (i + 1)
        circle = Circle((x, node_y), 0.15, facecolor=color, edgecolor='#333333', linewidth=1)
        ax.add_patch(circle)
        nodes.append((x, node_y))

        # Add node label if provided
        if node_labels and i < len(node_labels):
            ax.text(x + 0.3, node_y, node_labels[i], ha='left', va='center', fontsize=8)

    if label:
        ax.text(x, y - 0.2, label, ha='center', va='top', fontsize=8, fontweight='bold')

    return nodes

def draw_connections(ax, layer1, layer2, color='#BDBDBD', alpha=0.3):
    """Draw connections between two layers of nodes."""
    for n1 in layer1:
        for n2 in layer2:
            ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color=color, alpha=alpha, linewidth=0.5)

# ============== DRAW THE ARCHITECTURE ==============

# Title
ax.text(10, 7.7, 'Multi-Task Learning CNN Architecture',
        ha='center', va='top', fontsize=14, fontweight='bold', color=colors['text'])

# ---------- INPUT ----------
# Draw input image representation
input_img = FancyBboxPatch((0.3, 3), 1.2, 1.2, boxstyle="round,pad=0.02",
                            facecolor=colors['input'], edgecolor='#1565C0', linewidth=1.5)
ax.add_patch(input_img)

# Add grid lines to represent image
for i in range(5):
    ax.plot([0.3 + i*0.3, 0.3 + i*0.3], [3, 4.2], color='#90CAF9', linewidth=0.5, alpha=0.5)
    ax.plot([0.3, 1.5], [3 + i*0.3, 3 + i*0.3], color='#90CAF9', linewidth=0.5, alpha=0.5)

ax.text(0.9, 4.6, 'Input', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(0.9, 2.7, '(32 × 32 × 1)', ha='center', va='top', fontsize=8, style='italic')

# Arrow from input
draw_arrow(ax, (1.6, 3.6), (2.0, 3.6))

# ---------- SHARED BACKBONE ----------
ax.text(3.5, 6.2, 'SHARED BACKBONE', ha='center', va='bottom', fontsize=11,
        fontweight='bold', color='#2E7D32',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32'))

# Conv Block 1
draw_3d_block(ax, 2.1, 2.8, 0.9, 1.6, 32, colors['conv_shared'],
              label='Conv2D', sublabel='32 filters, 3×3')
ax.text(2.7, 5.5, 'MaxPool\n(16×16)', ha='center', va='bottom', fontsize=7)
draw_arrow(ax, (3.3, 3.6), (3.7, 3.6))

# Conv Block 2
draw_3d_block(ax, 3.8, 2.9, 0.8, 1.4, 64, colors['conv_shared'],
              label='Conv2D', sublabel='64 filters, 3×3')
ax.text(4.4, 5.4, 'MaxPool\n(8×8)', ha='center', va='bottom', fontsize=7)
draw_arrow(ax, (5.0, 3.6), (5.4, 3.6))

# Conv Block 3
draw_3d_block(ax, 5.5, 3.0, 0.7, 1.2, 128, colors['conv_shared'],
              label='Conv2D', sublabel='128 filters, 3×3')

# Branching point
ax.plot([6.6, 7.2], [3.6, 3.6], color=colors['arrow'], linewidth=2)
ax.plot([7.2, 7.2], [2.0, 5.2], color=colors['arrow'], linewidth=2)

# ---------- TASK A BRANCH (Top) ----------
ax.text(9.5, 6.8, 'Task A: Shape Classification (10 classes)',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#D84315')

draw_arrow(ax, (7.2, 5.2), (7.8, 5.2))
draw_3d_block(ax, 7.9, 4.6, 0.5, 1.0, 128, colors['conv_a'], sublabel='Conv 128')
draw_arrow(ax, (8.7, 5.1), (9.1, 5.1))
draw_3d_block(ax, 9.2, 4.6, 0.5, 1.0, 128, colors['conv_a'], sublabel='Conv 128')
draw_arrow(ax, (10.0, 5.1), (10.4, 5.1))

ax.text(10.7, 5.5, 'GAP', ha='center', va='bottom', fontsize=8, fontweight='bold')
rect_gap_a = FancyBboxPatch((10.5, 4.8), 0.4, 0.6, boxstyle="round",
                             facecolor=colors['conv_a'], edgecolor='#BF360C', linewidth=1)
ax.add_patch(rect_gap_a)

draw_arrow(ax, (11.0, 5.1), (11.4, 5.1))

# Dense layer for A
nodes_a_dense = draw_neural_layer(ax, 11.7, 4.4, 4, 1.4, colors['conv_a'])
ax.text(11.7, 6.0, 'Dense\n64', ha='center', va='bottom', fontsize=8)

draw_arrow(ax, (12.0, 5.1), (12.5, 5.1))

# Output A
nodes_a_out = draw_neural_layer(ax, 12.8, 4.2, 5, 1.8, colors['output_a'],
                                 node_labels=['0', '1', '...', '8', '9'])
ax.text(12.8, 6.3, 'Output A\nSoftmax', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Connections
draw_connections(ax, nodes_a_dense, nodes_a_out, '#EF9A9A', 0.4)

# ---------- TASK B BRANCH (Middle) ----------
ax.text(10.2, 4.0, 'Task B: Orientation (32 classes)',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#6A1B9A')

draw_arrow(ax, (7.2, 3.6), (7.6, 3.6))
draw_3d_block(ax, 7.7, 3.0, 0.45, 0.95, 64, colors['conv_b'], sublabel='Conv 64')
draw_arrow(ax, (8.4, 3.5), (8.7, 3.5))
draw_3d_block(ax, 8.8, 3.0, 0.45, 0.95, 64, colors['conv_b'], sublabel='Conv 64')
draw_arrow(ax, (9.5, 3.5), (9.8, 3.5))
draw_3d_block(ax, 9.9, 3.0, 0.4, 0.85, 128, colors['conv_b'], sublabel='Conv 128')
draw_arrow(ax, (10.55, 3.4), (10.85, 3.4))

# MaxPool x2 for Task B
maxpool_b = FancyBboxPatch((10.9, 3.05), 0.5, 0.7, boxstyle="round",
                            facecolor='#E1BEE7', edgecolor='#6A1B9A', linewidth=1)
ax.add_patch(maxpool_b)
ax.text(11.15, 3.4, 'MaxPool\n×2', ha='center', va='center', fontsize=6, fontweight='bold')
draw_arrow(ax, (11.45, 3.4), (11.75, 3.4))

# Flatten for Task B
flatten_b = FancyBboxPatch((11.8, 3.1), 0.4, 0.6, boxstyle="round",
                            facecolor='#E1BEE7', edgecolor='#6A1B9A', linewidth=1)
ax.add_patch(flatten_b)
ax.text(12.0, 3.4, 'Flat', ha='center', va='center', fontsize=7)
draw_arrow(ax, (12.25, 3.4), (12.55, 3.4))

# Concatenate box
concat_box = FancyBboxPatch((12.6, 2.9), 0.7, 1.0, boxstyle="round",
                             facecolor='#FFE0B2', edgecolor='#E65100', linewidth=1.5)
ax.add_patch(concat_box)
ax.text(12.95, 3.4, 'Concat', ha='center', va='center', fontsize=7, fontweight='bold')

# Semantic transfer arrow from A to B (dashed)
arrow_transfer = FancyArrowPatch((11.7, 4.8), (12.95, 3.95),
                                  arrowstyle='->', mutation_scale=12,
                                  connectionstyle='arc3,rad=-0.2',
                                  color='#D84315', linewidth=2, linestyle='--')
ax.add_patch(arrow_transfer)
ax.text(11.8, 4.3, 'A→B\nTransfer', ha='center', va='center', fontsize=7,
        color='#D84315', fontweight='bold')

draw_arrow(ax, (13.35, 3.4), (13.65, 3.4))

# Dense layer for B
nodes_b_dense = draw_neural_layer(ax, 13.95, 2.7, 5, 1.4, colors['conv_b'])
ax.text(13.95, 4.3, 'Dense\n256', ha='center', va='bottom', fontsize=8)

draw_arrow(ax, (14.25, 3.4), (14.55, 3.4))

# Output B
nodes_b_out = draw_neural_layer(ax, 14.85, 2.5, 6, 1.8, colors['output_b'],
                                 node_labels=['0', '1', '...', '30', '31'])
ax.text(14.85, 4.6, 'Output B\nSoftmax', ha='center', va='bottom', fontsize=8, fontweight='bold')

draw_connections(ax, nodes_b_dense, nodes_b_out, '#CE93D8', 0.4)

# ---------- TASK C BRANCH (Bottom) ----------
ax.text(9.5, 1.2, 'Task C: Regression',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#F57F17')

draw_arrow(ax, (7.2, 2.0), (7.8, 2.0))

# StopGradient layer (special octagon shape)
stop_grad = FancyBboxPatch((7.9, 1.5), 1.0, 0.9, boxstyle="round",
                            facecolor='#FFCDD2', edgecolor='#C62828', linewidth=2)
ax.add_patch(stop_grad)
ax.text(8.4, 1.95, 'STOP\nGRAD', ha='center', va='center', fontsize=7,
        fontweight='bold', color='#C62828')

draw_arrow(ax, (9.0, 2.0), (9.4, 2.0))

# GAP for C
rect_gap_c = FancyBboxPatch((9.5, 1.6), 0.5, 0.7, boxstyle="round",
                             facecolor=colors['conv_c'], edgecolor='#F57F17', linewidth=1)
ax.add_patch(rect_gap_c)
ax.text(9.75, 2.5, 'GAP', ha='center', va='bottom', fontsize=8)

draw_arrow(ax, (10.1, 2.0), (10.5, 2.0))

# Dense for C
nodes_c_dense = draw_neural_layer(ax, 10.8, 1.3, 3, 1.2, colors['conv_c'])
ax.text(10.8, 2.7, 'Dense\n32', ha='center', va='bottom', fontsize=8)

draw_arrow(ax, (11.1, 1.9), (11.5, 1.9))

# Output C (single node for regression)
circle_out = Circle((11.8, 1.9), 0.2, facecolor=colors['output_c'],
                     edgecolor='#E65100', linewidth=2)
ax.add_patch(circle_out)
ax.text(11.8, 2.4, 'Output C\nSigmoid', ha='center', va='bottom', fontsize=8, fontweight='bold')

draw_connections(ax, nodes_c_dense, [(11.8, 1.9)], '#FFE082', 0.5)

# ---------- LEGEND ----------
legend_x = 16.0
legend_y = 5.5

ax.text(legend_x + 1, 6.5, 'Legend', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Legend items
legend_items = [
    (colors['conv_shared'], 'Shared Backbone'),
    (colors['conv_a'], 'Task A Head'),
    (colors['conv_b'], 'Task B Head'),
    (colors['conv_c'], 'Task C Head'),
    ('#FFCDD2', 'Gradient Isolation'),
]

for i, (color, label) in enumerate(legend_items):
    rect = FancyBboxPatch((legend_x, legend_y - i*0.5), 0.4, 0.3,
                           boxstyle="round", facecolor=color, edgecolor='#555555')
    ax.add_patch(rect)
    ax.text(legend_x + 0.6, legend_y - i*0.5 + 0.15, label,
            ha='left', va='center', fontsize=8)

# Dashed line for transfer
ax.plot([legend_x, legend_x + 0.4], [legend_y - 2.5, legend_y - 2.5],
        color='#D84315', linewidth=2, linestyle='--')
ax.text(legend_x + 0.6, legend_y - 2.5, 'Semantic Transfer',
        ha='left', va='center', fontsize=8)

# ---------- ANNOTATIONS ----------
# Model stats
ax.text(17.5, 2.5, 'Model Stats:', ha='left', va='top', fontsize=9, fontweight='bold')
ax.text(17.5, 2.1, '• ~200K parameters', ha='left', va='top', fontsize=8)
ax.text(17.5, 1.7, '• Hard parameter sharing', ha='left', va='top', fontsize=8)
ax.text(17.5, 1.3, '• 3 task-specific heads', ha='left', va='top', fontsize=8)

# Save
plt.tight_layout()
plt.savefig('images/architecture_diagram_pro.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('architecture_diagram_pro.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ Saved: images/architecture_diagram_pro.png")
print("✅ Saved: architecture_diagram_pro.png")

plt.close()
