import numpy as np
from src.network import FCNN
from src.dataloader import load_dataset, load_nls_dataset, load_regression_dataset
from src.metrics import compute_classification_metrics, compute_regression_metrics
from src.plots import plot_loss, plot_decision_region, plot_3d_activations, plot_regression_scatter, plot_regression_overlay


def run_classification_experiment(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, hidden_architectures, dataset_name):

    print(f"Running Classification: {dataset_name}")
    

    best_val_f_measure = -1
    best_model = None
    best_arch = None
    best_train_loss = None
    best_val_loss = None

    for hidden_nodes in hidden_architectures:
        arch = [X_train.shape[1]] + hidden_nodes + [num_classes]
        print(f"\n--- Training Architecture: {arch} ---")

        net = FCNN(arch, output_activation="logistic")
        train_loss, val_loss = net.fit(X_train, y_train, X_val, y_val, epochs=500, lr=0.01, verbose=False)

        val_preds_raw = net.predict(X_val)
        y_val_labels = np.argmax(y_val, axis=1)

        metrics = compute_classification_metrics(y_val_labels, val_preds_raw, num_classes)
        print("Validation Metrics for this architecture:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (Per-Class): {np.round(metrics['precision'], 4)} | Mean: {metrics['mean_precision']:.4f}")
        print(f"  Recall (Per-Class):    {np.round(metrics['recall'], 4)} | Mean: {metrics['mean_recall']:.4f}")
        print(f"  F-Measure (Per-Class): {np.round(metrics['f_measure'], 4)} | Mean: {metrics['mean_f_measure']:.4f}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

        if metrics['mean_f_measure'] > best_val_f_measure:
            best_val_f_measure = metrics['mean_f_measure']
            best_model = net
            best_arch = arch
            best_train_loss = train_loss
            best_val_loss = val_loss

    print(f"\n*** BEST ARCHITECTURE SELECTED: {best_arch} ***")

    # Evaluate best model on Test Data
    test_preds_raw = best_model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    test_metrics = compute_classification_metrics(y_test_labels, test_preds_raw, num_classes)

    print("\n--- BEST MODEL TEST METRICS ---")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision (Per-Class): {np.round(test_metrics['precision'], 4)} | Mean: {test_metrics['mean_precision']:.4f}")
    print(f"  Recall (Per-Class):    {np.round(test_metrics['recall'], 4)} | Mean: {test_metrics['mean_recall']:.4f}")
    print(f"  F-Measure (Per-Class): {np.round(test_metrics['f_measure'], 4)} | Mean: {test_metrics['mean_f_measure']:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Generate Plots for the best model
    print("\nGenerating Plots...")
    plot_loss(best_train_loss, best_val_loss, title=f"{dataset_name} - Error vs Epochs")
    plot_decision_region(X_train, y_train, best_model, title=f"{dataset_name} - Decision Region")

    # 3D Node plots for Train, Val, Test for all hidden and output nodes
    splits = [("Train", X_train), ("Val", X_val), ("Test", X_test)]
    for split_name, X_split in splits:
        for i in range(1, len(best_model.weights) + 1):
            layer_name = f"Hidden Layer {i}" if i < len(best_model.weights) else "Output Layer"
            plot_3d_activations(best_model, X_split, layer_idx=i, title=f"{dataset_name} - {split_name} - {layer_name}")

    return best_model


def run_regression_experiment(X_train, y_train, X_val, y_val, X_test, y_test, hidden_architectures, dataset_name):

    print(f"Running Regression: {dataset_name}")


    best_val_rmse = float('inf')
    best_model = None
    best_arch = None
    best_train_loss = None
    best_val_loss = None

    for hidden_nodes in hidden_architectures:
        arch = [X_train.shape[1]] + hidden_nodes + [1]
        print(f"\n--- Training Architecture: {arch} ---")

        net = FCNN(arch, output_activation="linear")
        train_loss, val_loss = net.fit(X_train, y_train, X_val, y_val, epochs=1000, lr=0.001, verbose=False)

        # Print Train AND Val RMSE for all architectures
        train_preds = net.predict(X_train)
        train_metrics = compute_regression_metrics(y_train, train_preds)
        print(f"  Train RMSE: {train_metrics['rmse']:.4f} | %RMSE: {train_metrics['pct_rmse']:.2f}%")

        val_preds = net.predict(X_val)
        val_metrics = compute_regression_metrics(y_val, val_preds)
        print(f"  Val RMSE:   {val_metrics['rmse']:.4f} | %RMSE: {val_metrics['pct_rmse']:.2f}%")

        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_model = net
            best_arch = arch
            best_train_loss = train_loss
            best_val_loss = val_loss

    print(f"\n*** BEST ARCHITECTURE SELECTED: {best_arch} ***")

    # Evaluate best model on Test Data
    test_preds = best_model.predict(X_test)
    test_metrics = compute_regression_metrics(y_test, test_preds)

    print("\n--- BEST MODEL TEST METRICS ---")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f} | %RMSE: {test_metrics['pct_rmse']:.2f}%")

    # Generate Plots
    print("\nGenerating Plots...")
    plot_loss(best_train_loss, best_val_loss, title=f"{dataset_name} - Error vs Epochs")
    
    splits = [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]
    for split_name, X_split, y_split in splits:
        # Overlay plots
        plot_regression_overlay(X_split, y_split, best_model, title=f"{dataset_name} - {split_name} Overlay")
        # Scatter plots
        preds = best_model.predict(X_split)
        plot_regression_scatter(y_split, preds, title=f"{dataset_name} - {split_name} Target vs Output Scatter")
        # Node Activations
        for i in range(1, len(best_model.weights) + 1):
            layer_name = f"Hidden Layer {i}" if i < len(best_model.weights) else "Output Layer"
            plot_3d_activations(best_model, X_split, layer_idx=i, title=f"{dataset_name} - {split_name} - {layer_name}")

    return best_model


if __name__ == "__main__":
    class1_files = [
        "Data/Classification/LS_Group26/Class1.txt", 
        "Data/Classification/LS_Group26/Class2.txt", 
        "Data/Classification/LS_Group26/Class3.txt"
    ]
    X_train_c1, y_train_c1, X_val_c1, y_val_c1, X_test_c1, y_test_c1 = load_dataset(class1_files)
    run_classification_experiment(X_train_c1, y_train_c1, X_val_c1, y_val_c1, X_test_c1, y_test_c1, 3, [[4], [8], [16]], "Classification Dataset 1")

    counts = [500, 500, 1000] 
    X_train_c2, y_train_c2, X_val_c2, y_val_c2, X_test_c2, y_test_c2 = load_nls_dataset("Data/Classification/NLS_Group26.txt", counts)
    run_classification_experiment(X_train_c2, y_train_c2, X_val_c2, y_val_c2, X_test_c2, y_test_c2, len(counts), [[4, 4], [8, 4], [16, 8]], "Classification Dataset 2")

    X_train_r1, y_train_r1, X_val_r1, y_val_r1, X_test_r1, y_test_r1 = load_regression_dataset("Data/Regression/UnivariateData/26.csv")
    run_regression_experiment(X_train_r1, y_train_r1, X_val_r1, y_val_r1, X_test_r1, y_test_r1, [[4], [8], [16]], "Regression Dataset 1")

    X_train_r2, y_train_r2, X_val_r2, y_val_r2, X_test_r2, y_test_r2 = load_regression_dataset("Data/Regression/BivariateData/26.csv")
    run_regression_experiment(X_train_r2, y_train_r2, X_val_r2, y_val_r2, X_test_r2, y_test_r2, [[8], [16], [8, 4], [16, 8]], "Regression Dataset 2")
