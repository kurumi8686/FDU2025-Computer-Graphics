import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import os  # 新增os模块

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.gaussian_process')
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Predicted variances smaller than 0. Setting to 0.")


def load_data(file_path):
    inputs = []
    outputs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data_point = json.loads(line)
                    if "input" in data_point and "output" in data_point:
                        inputs.append(data_point["input"])
                        outputs.append(data_point["output"])
                    else:
                        print(f"Skipping line due to missing 'input' or 'output': {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
        if not inputs or not outputs:
            raise ValueError("No valid data loaded. Check file format and content.")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    return inputs, outputs


# 修改 get_embeddings 函数以接受本地路径
def get_embeddings(texts, model_path_or_name='all-MiniLM-L6-v2'):
    print(f"Loading sentence transformer model from: {model_path_or_name}...")
    # 检查路径是否存在，如果存在则认为是本地路径
    if os.path.isdir(model_path_or_name):
        print("Attempting to load model from local path.")
    else:
        print("Attempting to load model from Hugging Face Hub (this may fail if offline).")
    try:
        model = SentenceTransformer(model_path_or_name)
    except Exception as e:
        print(f"Error loading sentence transformer model: {e}")
        print("If you are offline and intended to use a local model, ensure:")
        print(f"1. The path '{model_path_or_name}' is correct.")
        print("2. The directory contains all necessary model files.")
        raise
    print("Embedding texts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def main():
    # --- 参数配置 ---
    file_path = 'data_test.jsonl'
    n_train_samples_user_choice = 1000
    dim_input_pca_for_gpr = 2
    dim_output_pca_visual = 2
    local_model_directory = '/root/models/all-MiniLM-L6-v2/all-MiniLM-L6-v2'

    # --- 1. 加载数据 ---
    print("Loading data...")
    try:
        input_texts, output_codes = load_data(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    num_total_samples = len(input_texts)
    print(f"Successfully loaded {num_total_samples} data points.")

    # --- 2. 调整参数以适应数据大小 ---
    n_train_samples = min(n_train_samples_user_choice, num_total_samples)
    if n_train_samples < 1:
        print("Number of training samples must be at least 1. Setting to 1.")
        n_train_samples = 1
    if n_train_samples_user_choice > num_total_samples:
        print(
            f"Warning: Requested {n_train_samples_user_choice} training samples, but only {num_total_samples} available. Using {n_train_samples} samples.")

    actual_dim_input_pca_for_gpr = min(dim_input_pca_for_gpr, num_total_samples - 1 if num_total_samples > 1 else 1)
    actual_dim_input_pca_for_gpr = max(1, actual_dim_input_pca_for_gpr)
    if actual_dim_input_pca_for_gpr != dim_input_pca_for_gpr:
        print(f"Adjusted PCA components for GPR input to: {actual_dim_input_pca_for_gpr}")
    if dim_output_pca_visual >= num_total_samples and num_total_samples > 0:
        print(f"Warning: PCA components for output ({dim_output_pca_visual}) is >= number of samples ({num_total_samples}). Adjusting to {max(1, num_total_samples - 1)}.")
        dim_output_pca_visual = max(1, num_total_samples - 1 if num_total_samples > 1 else 1)
    elif dim_output_pca_visual <= 0:
        dim_output_pca_visual = 1

    # --- 3. 获取嵌入 ---
    model_source = local_model_directory
    if not local_model_directory or not os.path.isdir(local_model_directory):
        print(f"Local model directory '{local_model_directory}' not found or not specified. Attempting to use default 'all-MiniLM-L6-v2' from Hub.")
        model_source = 'all-MiniLM-L6-v2'
    try:
        input_embeddings = get_embeddings(input_texts, model_path_or_name=model_source)
        output_embeddings = get_embeddings(output_codes, model_path_or_name=model_source)
    except Exception as e:
        return

    # --- 4. 标准化嵌入 (可选但推荐用于PCA和GPR) ---
    scaler_input = StandardScaler()
    input_embeddings_scaled = scaler_input.fit_transform(input_embeddings)

    scaler_output = StandardScaler()
    output_embeddings_scaled = scaler_output.fit_transform(output_embeddings)

    # --- 5. PCA降维 ---
    print(f"Performing PCA on input embeddings to {actual_dim_input_pca_for_gpr} dimensions for GPR input...")
    pca_input = PCA(n_components=actual_dim_input_pca_for_gpr, random_state=42)
    X_reduced_for_gpr = pca_input.fit_transform(input_embeddings_scaled)

    print(
        f"Performing PCA on output embeddings to {dim_output_pca_visual} dimensions for GPR target and visualization...")
    pca_output_visual = PCA(n_components=dim_output_pca_visual, random_state=42)
    Y_reduced_visual_actual = pca_output_visual.fit_transform(output_embeddings_scaled)

    # --- 6. 准备GPR的训练和测试数据 ---
    X_gpr_train = X_reduced_for_gpr[:n_train_samples]
    Y_gpr_train_target = Y_reduced_visual_actual[:n_train_samples]

    # --- 7. 高斯过程回归 ---
    print(f"Training Gaussian Process Regressor on {n_train_samples} samples...")
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-5, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-5)

    try:
        gpr.fit(X_gpr_train, Y_gpr_train_target)
    except Exception as e:
        print(f"Error during GPR fitting: {e}")
        print("This can happen with very few training samples or ill-conditioned kernels.")
        print("Consider increasing n_train_samples or adjusting GPR/PCA parameters.")
        return

    print("Predicting with GPR for all input samples...")
    Y_reduced_visual_predicted, Y_std_predicted = gpr.predict(X_reduced_for_gpr, return_std=True)

    # --- 8. 可视化 ---
    print("Generating plot...")
    plt.figure(figsize=(12, 10))

    if dim_output_pca_visual == 1:
        plt.scatter(Y_reduced_visual_actual[:n_train_samples, 0],
                    np.zeros_like(Y_reduced_visual_actual[:n_train_samples, 0]) + 0.05,
                    label=f'Actual Output Embeddings (Train, {n_train_samples} points) - PCA Dim 1',
                    color='blue', marker='o', s=100, alpha=0.7, edgecolor='k')
        if n_train_samples < num_total_samples:
            plt.scatter(Y_reduced_visual_actual[n_train_samples:, 0],
                        np.zeros_like(Y_reduced_visual_actual[n_train_samples:, 0]) + 0.05,
                        label='Actual Output Embeddings (Test) - PCA Dim 1',
                        color='green', marker='o', s=100, alpha=0.7, edgecolor='k')
        plt.scatter(Y_reduced_visual_predicted[:, 0], np.zeros_like(Y_reduced_visual_predicted[:, 0]) - 0.05,
                    label='Predicted Output Embeddings (GPR) - PCA Dim 1',
                    color='red', marker='x', s=100, alpha=0.7)
        plt.yticks([])
        plt.xlabel(f'PCA Component 1 of Output Embedding')

    elif dim_output_pca_visual == 2:
        plt.scatter(Y_reduced_visual_actual[:n_train_samples, 0], Y_reduced_visual_actual[:n_train_samples, 1],
                    label=f'Actual Output Embeddings (Train, {n_train_samples} points)',
                    color='blue', marker='o', s=100, alpha=0.7, edgecolor='k')
        if n_train_samples < num_total_samples:
            plt.scatter(Y_reduced_visual_actual[n_train_samples:, 0], Y_reduced_visual_actual[n_train_samples:, 1],
                        label='Actual Output Embeddings (Test)',
                        color='green', marker='o', s=100, alpha=0.7, edgecolor='k')

        plt.scatter(Y_reduced_visual_predicted[:, 0], Y_reduced_visual_predicted[:, 1],
                    label='Predicted Output Embeddings (GPR)',
                    color='red', marker='x', s=100, alpha=0.7)

        plt.xlabel('PCA Component 1 of Output Embedding')
        plt.ylabel('PCA Component 2 of Output Embedding')
    else:
        print("Output PCA dimension is > 2. Plotting first two dimensions if available.")
        if dim_output_pca_visual >= 2:
            plt.scatter(Y_reduced_visual_actual[:n_train_samples, 0], Y_reduced_visual_actual[:n_train_samples, 1],
                        label=f'Actual Output Embeddings (Train, {n_train_samples} points) - First 2D',
                        color='blue', marker='o', s=100, alpha=0.7, edgecolor='k')
            if n_train_samples < num_total_samples:
                plt.scatter(Y_reduced_visual_actual[n_train_samples:, 0], Y_reduced_visual_actual[n_train_samples:, 1],
                            label='Actual Output Embeddings (Test) - First 2D',
                            color='green', marker='o', s=100, alpha=0.7, edgecolor='k')

            plt.scatter(Y_reduced_visual_predicted[:, 0], Y_reduced_visual_predicted[:, 1],
                        label='Predicted Output Embeddings (GPR) - First 2D',
                        color='red', marker='x', s=100, alpha=0.7)
            plt.xlabel('PCA Component 1 of Output Embedding')
            plt.ylabel('PCA Component 2 of Output Embedding')
        else:
            print("Cannot visualize: Output PCA dimension is less than 1 after adjustments.")

    plt.title(
        f'Actual vs. GPR Predicted Output Embeddings (PCA to {dim_output_pca_visual}D)\nInput Text Embeddings PCA to {actual_dim_input_pca_for_gpr}D for GPR Input')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()

    plot_filename = f"gpr_embedding_visualization_n{n_train_samples}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.show()


if __name__ == '__main__':
    main()