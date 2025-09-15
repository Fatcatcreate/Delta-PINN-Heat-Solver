import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python inspectNpy.py <path_to_npy_file>")
    sys.exit(1)

filepath = sys.argv[1]

print(f"Inspecting file: {filepath}")
try:
    data_array = np.load(filepath, allow_pickle=True)
    print(f"Successfully loaded {filepath}")
    
    print(f"Loaded array shape: {data_array.shape}")
    print(f"Loaded array dtype: {data_array.dtype}")

    if data_array.shape == ():
        print("Array is a 0-dimensional array, calling .item() to extract the object.")
        data = data_array.item()
        print("Object type after .item():", type(data))
        
        if isinstance(data, dict):
            print("File contains a dictionary with keys:", data.keys())
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  - Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}")
                else:
                    print(f"  - Key: {key}, Value: {value}")
        else:
            print("Object is not a dictionary.")
    else:
        print("Array is not a 0-dimensional array.")

except Exception as e:
    print(f"Error loading or inspecting {filepath}: {e}")