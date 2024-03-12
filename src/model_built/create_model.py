from tensorflow.keras import layers
from Mlops.src.utils.constant import df
import tensorflow as tf

num_features = df.shape[1] - 1
def create_model():
    """
    Creates a custom model for tabular data.

    This function defines a custom TensorFlow Keras Sequential model
    for tabular data with the following architecture:
    - Input layer with `num_features` neurons and 'relu' activation
    - Hidden layer with 64 neurons and 'relu' activation
    - Hidden layer with 32 neurons and 'relu' activation
    - Output layer with 1 neuron (for regression tasks)

    Returns:
        model (tf.keras.Model): The created Keras model.
    """
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(num_features,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model
