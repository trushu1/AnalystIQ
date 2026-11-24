from utils.data_processing import load_dataset

data = load_dataset("sample.csv")

print("Dataset loaded successfully!")
print(data.head())
