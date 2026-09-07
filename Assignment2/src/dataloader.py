import numpy as np  
import pandas as pd
def load_and_split(file_path):
    data = np.loadtxt(file_path)

    total_samples = len(data)
    
    train_end = int(0.60 * total_samples)
    val_end = int(0.80 * total_samples)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def create_label(index, num_samples):

    labels = np.zeros((num_samples,3))

    labels[:, index] = 1

    return labels



def load_dataset(class_file_path):
    all_X_train , all_y_train = [],[]
    all_X_val , all_y_val = [],[]
    all_X_test , all_y_test = [],[]

    for index, file_path in enumerate(class_file_path):
        
        #load and split using the previous function
        train_d , val_d , test_d = load_and_split(file_path)
        
        #Creating labels based on index
        X_train , y_train = train_d[:, :2], create_label(index,len(train_d))
        X_val, y_val = val_d[:, :2], create_label(index, len(val_d))
        X_test, y_test = test_d[:, :2], create_label(index, len(test_d))

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    

    #Concatenate all classes together
    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val = np.vstack(all_y_val)
    X_test = np.vstack(all_X_test)
    y_test = np.vstack(all_y_test)
    
    #Shuffle training data 
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    return X_train, y_train , X_val, y_val, X_test, y_test


def load_nls_dataset(file_path):
    #Skipping first line i.e(First 500 examples - class1; next 500 examples - class 2 and the last 1000 examples - class3.)
    data = np.loadtxt(file_path,skiprows = 1)

    class_blocks = [
        data[:500],     # Class 0
        data[500:1000], # Class 1
        data[1000:]     # Class 2
    ]

    all_X_train , all_y_train = [],[]
    all_X_val , all_y_val = [],[]
    all_X_test , all_y_test = [],[]

    for index, block in enumerate(class_blocks):
        total_samples = len(block)
        train_end = int(0.60 * total_samples)
        val_end = int(0.80 * total_samples)

        train_d = block[:train_end]
        val_d = block[train_end:val_end]
        test_d = block[val_end:]

        #Creating labels based on index
        X_train , y_train = train_d[:, :2], create_label(index,len(train_d))
        X_val, y_val = val_d[:, :2], create_label(index, len(val_d))
        X_test, y_test = test_d[:, :2], create_label(index, len(test_d))

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_X_val.append(X_val)
        all_y_val.append(y_val)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
    

    #Concatenate all classes together
    X_train = np.vstack(all_X_train)
    y_train = np.vstack(all_y_train)
    X_val = np.vstack(all_X_val)
    y_val = np.vstack(all_y_val)
    X_test = np.vstack(all_X_test)
    y_test = np.vstack(all_y_test)
    
    #Shuffle training data 
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    return X_train, y_train , X_val, y_val, X_test, y_test




def load_regression_dataset(file_path):
    df = pd.read_csv(file_path,header=None)
    data = df.to_numpy()

    np.random.shuffle(data)
    
    total_samples = len(data)
    
    train_end = int(0.60 * total_samples)
    val_end = int(0.80 * total_samples)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    X_train , y_train = train_data[:, :-1], train_data[:, -1:]
    X_val , y_val = val_data[:, :-1], val_data[:, -1:]
    X_test , y_test = test_data[:, :-1], test_data[:, -1:]
    
    return X_train , y_train , X_val , y_val , X_test , y_test





