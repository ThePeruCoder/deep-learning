import numpy as np
import pandas as pd
import os

# Set a fixed random seed so all random shuffles are deterministic
np.random.seed(42)

def split_and_save(X, y, name_prefix, is_stratified=True):
    """
    Helper function to split 70/30 and save NumPy binary files (.npy)
    """
    if is_stratified:
        # For Classification: Keep equal percentage of each class in train/test
        train_X_list, test_X_list = [], []
        train_y_list, test_y_list = [], []
        
        unique_classes = np.unique(y)
        for c in unique_classes:
            indices = np.where(y == c)[0]
            np.random.shuffle(indices)
            
            split_idx = int(len(indices) * 0.7)
            train_idx, test_idx = indices[:split_idx], indices[split_idx:]
            
            train_X_list.append(X[train_idx])
            test_X_list.append(X[test_idx])
            train_y_list.append(y[train_idx])
            test_y_list.append(y[test_idx])
            
        train_X = np.vstack(train_X_list)
        test_X = np.vstack(test_X_list)
        train_y = np.vstack(train_y_list)
        test_y = np.vstack(test_y_list)
    else:
        # For Regression: Standard 70/30 shuffle split
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * 0.7)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        train_X, test_X = X[train_idx], X[test_idx]
        train_y, test_y = y[train_idx], y[test_idx]

    # Save to data folder
    output_dir = os.path.dirname(name_prefix)
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f"{name_prefix}_train_X.npy", train_X)
    np.save(f"{name_prefix}_train_y.npy", train_y)
    np.save(f"{name_prefix}_test_X.npy", test_X)
    np.save(f"{name_prefix}_test_y.npy", test_y)
    print(f"Saved splits for: {name_prefix}")

# --- 1. Linearly Separable (LS) Classification ---
def process_ls():
    data_list, labels_list = [], []
    for class_id in range(1, 4):
        path = f"data/Classification/LS_Group26/class{class_id}.txt"
        points = np.loadtxt(path)
        labels = np.full((points.shape[0], 1), class_id - 1) # Labels: 0, 1, 2
        data_list.append(points)
        labels_list.append(labels)
        
    X = np.vstack(data_list)
    y = np.vstack(labels_list)
    split_and_save(X, y, "data/Classification/LS", is_stratified=True)

# --- 2. Non-Linearly Separable (NLS) Classification ---
def process_nls():
    path = "data/Classification/NLS_Group26.txt"
    raw_data = np.loadtxt(path, skiprows=1)
    X = raw_data[:, :2]
    
    # 500 class 0, 500 class 1, 1000 class 2
    y = np.zeros((raw_data.shape[0], 1))
    y[500:1000] = 1
    y[1000:] = 2
    split_and_save(X, y, "data/Classification/NLS", is_stratified=True)

# --- 3. Univariate Regression (.csv) ---
def process_univariate():
    path = "data/Regression/UnivariateData/26.csv"
    # Using header=None in case the CSV doesn't have column headers
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, 0].values.reshape(-1, 1) # First column: input x
    y = df.iloc[:, 1].values.reshape(-1, 1) # Second column: target y
    split_and_save(X, y, "data/Regression/univariate", is_stratified=False)

# --- 4. Bivariate Regression (.csv) ---
def process_bivariate():
    path = "data/Regression/BivariateData/26.csv"
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :2].values                # First two columns: x1, x2
    y = df.iloc[:, 2].values.reshape(-1, 1)  # Third column: target y
    split_and_save(X, y, "data/Regression/bivariate", is_stratified=False)

if __name__ == "__main__":
    print("Generating exact 70/30 data splits for all datasets...")
    process_ls()
    process_nls()
    process_univariate()
    process_bivariate()
    print("Done! All .npy files are created successfully inside data/ folders.")