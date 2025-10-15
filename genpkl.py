import numpy as np
import pickle
import os

def generate_random_data(size_in_mb=30):
    # Approximate number of entries for 3MB (based on data structure and type)
    num_elements = (size_in_mb * 1024 ) // 8  # Each float64 element is 8 bytes
    num_columns = 100  # Create a dataset with 100 columns
    data = np.random.rand(num_elements, num_columns)  # Random data with values between 0 and 1
    
    # For realism, use some basic data structure that can be stored in a pickle file
    data_dict = {
        'data': data,
        'metadata': {
            'num_rows': num_elements,
            'num_columns': num_columns,
            'generated_at': '2025-10-15'
        }
    }
    
    return data_dict

def save_to_pkl(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Pickle file '{filename}' created with size {os.path.getsize(filename) / (1024 * 1024):.2f} MB.")

if __name__ == "__main__":
    pkl_filename = 'random_data.pkl'
    data = generate_random_data(size_in_mb=3)
    save_to_pkl(pkl_filename, data)
