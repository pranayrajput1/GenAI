from Mlops.src.utils.constant import df
from Mlops.src.model_built.create_model import create_model

model = create_model()



num_features = df.shape[1] - 1

def train_model(model, X, y, model_path):
    """
    Trains the provided model on the given features (X) and target (y).

    Args:
        model (tf.keras.Model): The Keras model to train.
        X (pd.DataFrame): The DataFrame containing input features.
        y (pd.Series): The target variable.
        model_path (str): The path to save the trained model.

    This function performs the following steps:
    1. Compiles the model with 'adam' optimizer, 'mean_squared_error' loss, and metrics.
    2. Trains the model on features (X) and target (y) for 10 epochs with 20% validation split.
    3. Saves the trained model to the specified `model_path`.
    """
    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    # Train the model
    model.fit(X, y, epochs=10, validation_split=0.2)

    # Save the trained model
    model.save(model_path)




