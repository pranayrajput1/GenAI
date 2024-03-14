import pathlib

path = pathlib.Path(__file__).resolve().parent.parent
dataset_path = path / "test_data"
file_path = dataset_path / "household_power_consumption .csv"
household_train_test_data_path = dataset_path/ "testing_data.csv"
plot_data_path = dataset_path/ "plot_data.csv"
train_data_path = dataset_path/ "train_data.csv"
batches_path = dataset_path/ "train_batches.csv"
trained_model_path = dataset_path/ "db_scan.joblib"