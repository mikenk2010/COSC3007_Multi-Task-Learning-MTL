#!/usr/bin/env python3
"""
Create MTL architecture diagram using pydot directly (no TensorFlow).
"""
import pydot
import os

os.makedirs('images', exist_ok=True)

# Create graph
graph = pydot.Dot(graph_type='digraph', rankdir='TB', bgcolor='white', dpi='150')
graph.set_node_defaults(shape='box', style='filled', fontname='Helvetica', fontsize='10')

# Color scheme
colors = {
    'input': '#E8F5E9',
    'shared': '#BBDEFB',
    'task_a': '#FFCCBC',
    'task_b': '#D1C4E9',
    'task_c': '#FFF9C4',
    'output': '#F5F5F5'
}

# Input layer
graph.add_node(pydot.Node('input', label='InputLayer\n(32, 32, 1)', fillcolor=colors['input']))

# Shared Backbone
graph.add_node(pydot.Node('conv1', label='Conv2D (32 filters)\n3×3, ReLU', fillcolor=colors['shared']))
graph.add_node(pydot.Node('pool1', label='MaxPooling2D\n(2×2) → 16×16', fillcolor=colors['shared']))
graph.add_node(pydot.Node('conv2', label='Conv2D (64 filters)\n3×3, ReLU', fillcolor=colors['shared']))
graph.add_node(pydot.Node('pool2', label='MaxPooling2D\n(2×2) → 8×8', fillcolor=colors['shared']))
graph.add_node(pydot.Node('conv3', label='Conv2D (128 filters)\n3×3, ReLU', fillcolor=colors['shared']))

# Task A Branch
graph.add_node(pydot.Node('a_conv1', label='Conv2D (128)\nReLU', fillcolor=colors['task_a']))
graph.add_node(pydot.Node('a_conv2', label='Conv2D (128)\nReLU', fillcolor=colors['task_a']))
graph.add_node(pydot.Node('a_gap', label='GlobalAvgPool2D', fillcolor=colors['task_a']))
graph.add_node(pydot.Node('a_dense', label='Dense (64)\nReLU', fillcolor=colors['task_a']))
graph.add_node(pydot.Node('a_drop', label='Dropout (0.5)', fillcolor=colors['task_a']))
graph.add_node(pydot.Node('output_a', label='output_A\nDense (10)\nSoftmax', fillcolor=colors['output'], shape='ellipse'))

# Task B Branch
graph.add_node(pydot.Node('b_conv1', label='Conv2D (64)\nReLU', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_conv2', label='Conv2D (64)\nReLU', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_conv3', label='Conv2D (128)\nReLU', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_pool1', label='MaxPool2D', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_pool2', label='MaxPool2D', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_flat', label='Flatten', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_concat', label='Concatenate\n[B features + A features]', fillcolor='#FFE0B2', shape='hexagon'))
graph.add_node(pydot.Node('b_dense', label='Dense (256)\nReLU', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('b_drop', label='Dropout (0.5)', fillcolor=colors['task_b']))
graph.add_node(pydot.Node('output_b', label='output_B\nDense (32)\nSoftmax', fillcolor=colors['output'], shape='ellipse'))

# Task C Branch
graph.add_node(pydot.Node('c_stop', label='StopGradient\n(blocks backward)', fillcolor='#FFCDD2', shape='octagon'))
graph.add_node(pydot.Node('c_gap', label='GlobalAvgPool2D', fillcolor=colors['task_c']))
graph.add_node(pydot.Node('c_dense', label='Dense (32)\nReLU', fillcolor=colors['task_c']))
graph.add_node(pydot.Node('c_drop', label='Dropout (0.3)', fillcolor=colors['task_c']))
graph.add_node(pydot.Node('output_c', label='output_C\nDense (1)\nSigmoid', fillcolor=colors['output'], shape='ellipse'))

# Add edges - Shared backbone
graph.add_edge(pydot.Edge('input', 'conv1'))
graph.add_edge(pydot.Edge('conv1', 'pool1'))
graph.add_edge(pydot.Edge('pool1', 'conv2'))
graph.add_edge(pydot.Edge('conv2', 'pool2'))
graph.add_edge(pydot.Edge('pool2', 'conv3'))

# Branch point (conv3 to three heads)
graph.add_edge(pydot.Edge('conv3', 'a_conv1', label='Task A', color='#E64A19'))
graph.add_edge(pydot.Edge('conv3', 'b_conv1', label='Task B', color='#7B1FA2'))
graph.add_edge(pydot.Edge('conv3', 'c_stop', label='Task C', color='#F9A825'))

# Task A flow
graph.add_edge(pydot.Edge('a_conv1', 'a_conv2'))
graph.add_edge(pydot.Edge('a_conv2', 'a_gap'))
graph.add_edge(pydot.Edge('a_gap', 'a_dense'))
graph.add_edge(pydot.Edge('a_dense', 'a_drop'))
graph.add_edge(pydot.Edge('a_drop', 'output_a'))

# Task B flow
graph.add_edge(pydot.Edge('b_conv1', 'b_conv2'))
graph.add_edge(pydot.Edge('b_conv2', 'b_conv3'))
graph.add_edge(pydot.Edge('b_conv3', 'b_pool1'))
graph.add_edge(pydot.Edge('b_pool1', 'b_pool2'))
graph.add_edge(pydot.Edge('b_pool2', 'b_flat'))
graph.add_edge(pydot.Edge('b_flat', 'b_concat'))
graph.add_edge(pydot.Edge('a_dense', 'b_concat', style='dashed', color='#E64A19', label='A→B\nTransfer'))  # Semantic transfer!
graph.add_edge(pydot.Edge('b_concat', 'b_dense'))
graph.add_edge(pydot.Edge('b_dense', 'b_drop'))
graph.add_edge(pydot.Edge('b_drop', 'output_b'))

# Task C flow
graph.add_edge(pydot.Edge('c_stop', 'c_gap'))
graph.add_edge(pydot.Edge('c_gap', 'c_dense'))
graph.add_edge(pydot.Edge('c_dense', 'c_drop'))
graph.add_edge(pydot.Edge('c_drop', 'output_c'))

# Add legend
legend = pydot.Cluster('legend', label='Legend', style='dashed', color='gray')
legend.add_node(pydot.Node('leg_shared', label='Shared Backbone', fillcolor=colors['shared']))
legend.add_node(pydot.Node('leg_a', label='Task A Head', fillcolor=colors['task_a']))
legend.add_node(pydot.Node('leg_b', label='Task B Head', fillcolor=colors['task_b']))
legend.add_node(pydot.Node('leg_c', label='Task C Head', fillcolor=colors['task_c']))
legend.add_node(pydot.Node('leg_stop', label='Gradient Isolation', fillcolor='#FFCDD2', shape='octagon'))
graph.add_subgraph(legend)

# Save
graph.write_png('images/architecture_diagram.png')
print("✅ Saved: images/architecture_diagram.png")

# Also save to root
graph.write_png('architecture_diagram.png')
print("✅ Saved: architecture_diagram.png")
