from utils.preprocessing import data_processing_pipeline

# Load and process the data
X_train, X_test, y_train, y_test = data_processing_pipeline('/home/nashtech/PycharmProjects/Mlops/Housing.csv')

# Now you can proceed with model training, e.g.:
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)