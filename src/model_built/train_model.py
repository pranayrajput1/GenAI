from Mlops.src.utils.constant import df
from Mlops.src.model_built.create_model import create_model
from Mlops.src.utils.constant import TARGET_COLUMN

model = create_model()



num_features = df.shape[1] - 1

def train_model(model, df, model_path):
    """
    Trains the provided model on the given DataFrame.

    Args:
        model (tf.keras.Model): The Keras model to train.
        df (pd.DataFrame): The DataFrame containing training data.
        model_path (str): The path to save the trained model.

    This function performs the following steps:
    1. Splits the DataFrame `df` into features (X) and target (y).
    2. Compiles the model with 'adam' optimizer, 'mean_squared_error' loss, and metrics.
    3. Trains the model on features (X) and target (y) for 10 epochs with 20% validation split.
    4. Saves the trained model to the specified `model_path`.

    Note:
    - `df` should have the target column named "SalePrice".
    - Ensure that the model has already been compiled before calling this function.

    Example:
        model = create_model()
        train_model(model, df_train, 'trained_model.h5')

    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])

    # Train the model
    model.fit(X, y, epochs=10, validation_split=0.2)

    # Save the trained model
    model.save(model_path)


