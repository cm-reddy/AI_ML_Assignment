from sklearn.datasets import fetch_california_housing
import pandas as pd

def load_and_save_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv("data/housing.csv", index=False)
    print("Dataset saved to data/housing.csv")

if __name__ == "__main__":
    load_and_save_data()
