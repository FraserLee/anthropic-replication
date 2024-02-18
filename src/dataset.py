import datasets
import os

def get_dataset():

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cache/dataset_cache'))
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

