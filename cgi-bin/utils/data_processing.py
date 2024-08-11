import pandas as pd

def process_input_data(file_path):
    data = pd.read_csv(file_path)
    processed_data = data.apply(lambda x: (x - x.mean()) / x.std())
    return processed_data
