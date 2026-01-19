from tensorflow.keras.utils import plot_model

# Assuming 'model' is your built model
plot_model(
    model,
    to_file='architecture_diagram.png',
    show_shapes=True,
    show_layer_names=False,
    rankdir='TB',  # Top-to-Bottom
    dpi=96
)