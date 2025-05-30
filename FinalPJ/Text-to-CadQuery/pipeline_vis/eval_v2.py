import os
import json
import shutil
import torch
import traceback
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm

# 确保脚本可以找到 assembly_pipeline_generate 模块
from assembly_pipeline_generate import (
    HF_MODEL_PATH, BLENDER_EXECUTABLE_PATH, BLENDER_SCRIPT_PATH,
    AutoTokenizer, AutoModelForCausalLM,
    assembly_text_to_image_pipeline
)

# 导入生成真值 STL 的函数
from generate_ground_truth_stls import generate_ground_truth_stls

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
INPUT_JSONL_FILE = os.path.join(PROJECT_ROOT, 'inference/test_filtered.jsonl')
PREDICTED_STL_DIR = os.path.join(PROJECT_ROOT, 'inference/output_assembly_pred_stl')
GROUND_TRUTH_STL_DIR = os.path.join(PROJECT_ROOT, 'inference/output_gt_from_jsonl')  # 新的真值目录

# --- Metric Calculation Functions (保持不变) ---
def sample_mesh_normalized(stl_path, num_points=10000):
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if not (mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0):
            return None
        mesh.apply_translation(-mesh.centroid)
        scale = np.max(mesh.bounding_box.extents)
        if scale == 0:
            return None
        mesh.apply_scale(1.0 / scale)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception as e:
        return None

def chamfer_distance(points1, points2):
    if points1 is None or points2 is None or points1.shape[0] == 0 or points2.shape[0] == 0:
        return np.nan
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer

def f1_score(points_pred, points_gt, threshold=0.02):
    if points_pred is None or points_gt is None or points_pred.shape[0] == 0 or points_gt.shape[0] == 0:
        return np.nan
    tree_pred = cKDTree(points_pred)
    tree_gt = cKDTree(points_gt)
    dist_pred_to_gt, _ = tree_gt.query(points_pred)
    precision = np.mean(dist_pred_to_gt < threshold)
    dist_gt_to_pred, _ = tree_pred.query(points_gt)
    recall = np.mean(dist_gt_to_pred < threshold)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def normalize_mesh(mesh):
    mesh = mesh.copy()
    if not (mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0):
        return mesh
    bounds = mesh.bounds
    scale_dims = bounds[1] - bounds[0]
    max_extent = np.max(scale_dims)
    if max_extent == 0:
        return mesh
    mesh.apply_translation(-bounds[0])
    mesh.apply_scale(1.0 / max_extent)
    return mesh

def voxelize_mesh(mesh, voxel_size=0.02):
    if not (mesh.vertices.shape[0] > 0 and mesh.faces.shape[0] > 0):
        return np.array([], dtype=bool)
    try:
        voxelized = mesh.voxelized(pitch=voxel_size)
        return voxelized.matrix
    except Exception as e:
        return np.array([], dtype=bool)

def volumetric_iou(mesh1_path, mesh2_path, voxel_size=0.02):
    try:
        mesh1 = trimesh.load(mesh1_path, force='mesh')
        mesh2 = trimesh.load(mesh2_path, force='mesh')

        if not (mesh1.vertices.shape[0] > 0 and mesh1.faces.shape[0] > 0 and \
                mesh2.vertices.shape[0] > 0 and mesh2.faces.shape[0] > 0):
            return np.nan

        mesh1_norm = normalize_mesh(mesh1)
        mesh2_norm = normalize_mesh(mesh2)
        
        vox1 = voxelize_mesh(mesh1_norm, voxel_size)
        vox2 = voxelize_mesh(mesh2_norm, voxel_size)

        if vox1.size == 0 or vox2.size == 0:
            return np.nan

        shape = np.maximum(vox1.shape, vox2.shape)
        vox1_padded = np.zeros(shape, dtype=bool)
        vox2_padded = np.zeros(shape, dtype=bool)
        
        if vox1.ndim == 3 and vox1.shape[0]>0 and vox1.shape[1]>0 and vox1.shape[2]>0:
            vox1_padded[:vox1.shape[0], :vox1.shape[1], :vox1.shape[2]] = vox1
        if vox2.ndim == 3 and vox2.shape[0]>0 and vox2.shape[1]>0 and vox2.shape[2]>0:
            vox2_padded[:vox2.shape[0], :vox2.shape[1], :vox2.shape[2]] = vox2
        
        intersection = np.logical_and(vox1_padded, vox2_padded).sum()
        union = np.logical_or(vox1_padded, vox2_padded).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        iou = intersection / union
        return iou
    except Exception as e:
        return np.nan

# --- 新增：生成真值 STL 函数 ---
def generate_ground_truth_stls_for_evaluation(max_samples: int = 10):
    """
    生成用于评估的真值 STL 文件
    """
    print("--- Step 0: Generating Ground Truth STLs from JSONL ---")
    
    if os.path.exists(GROUND_TRUTH_STL_DIR):
        existing_files = [f for f in os.listdir(GROUND_TRUTH_STL_DIR) if f.endswith('.stl')]
        if len(existing_files) >= max_samples:
            print(f"Ground truth directory already exists with {len(existing_files)} STL files.")
            print(f"Skipping generation. Delete {GROUND_TRUTH_STL_DIR} if you want to regenerate.")
            return len(existing_files)
    
    success_count, error_count = generate_ground_truth_stls(
        INPUT_JSONL_FILE, 
        GROUND_TRUTH_STL_DIR, 
        max_samples=max_samples
    )
    
    print(f"Generated {success_count} ground truth STL files.")
    return success_count

# --- 评估函数（修改后） ---
def generate_predicted_stls(model, tokenizer, max_samples: int = 10):
    """
    生成预测的 STL 文件
    """
    print("--- Step 1: Generating Predicted STLs using Assembly Pipeline ---")
    os.makedirs(PREDICTED_STL_DIR, exist_ok=True)

    print(f"Reading input prompts from: {INPUT_JSONL_FILE}")
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"Error: Input file {INPUT_JSONL_FILE} not found.")
        return 0

    prompts_to_process = []
    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_in:
        for line_number, line_content in enumerate(f_in):
            line_content = line_content.strip()
            if not line_content:
                continue
            try:
                data = json.loads(line_content)
                if "input" in data:
                    prompts_to_process.append(data["input"])
                else:
                    print(f"Skipping line {line_number}: 'input' key not found.")
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_number} due to JSON decoding error: {e}")
            
            if len(prompts_to_process) >= max_samples:
                print(f"Limited to first {max_samples} prompts for evaluation.")
                break

    print(f"Will process {len(prompts_to_process)} prompts.")

    generated_stl_count = 0

    for idx, user_prompt in enumerate(tqdm(prompts_to_process, desc="Generating Predicted STLs")):
        print(f"\nProcessing prompt {idx}: {user_prompt[:100]}...")
        
        target_pred_stl_path = os.path.join(PREDICTED_STL_DIR, f"{idx}.stl")
        dummy_image_output_path = os.path.join(PREDICTED_STL_DIR, f"temp_render_{idx}.png")

        try:
            _, generated_stl_path = assembly_text_to_image_pipeline(
                user_prompt=user_prompt,
                hf_model=model,
                hf_tokenizer=tokenizer,
                final_output_image_path=dummy_image_output_path,
                max_fix_attempts=2,
                eval_persistent_stl_path=target_pred_stl_path
            )

            if generated_stl_path and os.path.exists(generated_stl_path) and os.path.getsize(generated_stl_path) > 0:
                print(f"Successfully generated and saved STL to: {generated_stl_path}")
                generated_stl_count += 1
            else:
                print(f"Failed to generate STL for prompt {idx} or STL is empty.")

            if os.path.exists(dummy_image_output_path):
                os.remove(dummy_image_output_path)

        except Exception as e:
            print(f"Error processing prompt {idx} with assembly pipeline: {e}")
            traceback.print_exc()

    print(f"\n--- Finished generating {generated_stl_count} predicted STL files in {PREDICTED_STL_DIR} ---")
    return generated_stl_count

def compute_all_metrics():
    """
    计算所有指标
    """
    print("\n--- Step 2: Computing Metrics ---")
    if not os.path.exists(GROUND_TRUTH_STL_DIR):
        print(f"Error: Ground truth STL directory not found: {GROUND_TRUTH_STL_DIR}")
        return

    if not os.path.exists(PREDICTED_STL_DIR):
        print(f"Error: Predicted STL directory not found: {PREDICTED_STL_DIR}")
        return

    pred_files = {f for f in os.listdir(PREDICTED_STL_DIR) if f.endswith('.stl')}
    gt_files = {f for f in os.listdir(GROUND_TRUTH_STL_DIR) if f.endswith('.stl')}
    
    common_files = sorted(list(pred_files.intersection(gt_files)))

    if not common_files:
        print("No common STL files found between predicted and ground truth directories for comparison.")
        return

    print(f"Found {len(common_files)} common STL files for metric computation.")

    cd_results = []
    f1_results = []
    iou_results = []
    error_files_metric = []

    for filename in tqdm(common_files, desc="Computing Metrics"):
        gt_stl_path = os.path.join(GROUND_TRUTH_STL_DIR, filename)
        pred_stl_path = os.path.join(PREDICTED_STL_DIR, filename)

        # 检查预测 STL 是否存在和有效
        if not os.path.exists(pred_stl_path) or os.path.getsize(pred_stl_path) == 0:
            print(f"Skipping {filename}: Predicted STL missing or empty.")
            error_files_metric.append(filename + " (missing/empty predicted STL)")
            cd_results.append(np.nan)
            f1_results.append(np.nan)
            iou_results.append(np.nan)
            continue

        # 检查真值 STL 是否存在和有效
        if not os.path.exists(gt_stl_path) or os.path.getsize(gt_stl_path) == 0:
            print(f"Skipping {filename}: Ground truth STL missing or empty.")
            error_files_metric.append(filename + " (missing/empty ground truth STL)")
            cd_results.append(np.nan)
            f1_results.append(np.nan)
            iou_results.append(np.nan)
            continue

        points_gt = sample_mesh_normalized(gt_stl_path)
        points_pred = sample_mesh_normalized(pred_stl_path)

        if points_gt is None or points_pred is None:
            error_files_metric.append(filename + " (error sampling points)")
            cd_results.append(np.nan)
            f1_results.append(np.nan)
        else:
            cd = chamfer_distance(points_gt, points_pred)
            cd_results.append(cd)
            f1 = f1_score(points_pred, points_gt)
            f1_results.append(f1)
        
        iou = volumetric_iou(pred_stl_path, gt_stl_path)
        iou_results.append(iou)

    print("\n--- Metric Results ---")
    # 过滤掉 NaN 值
    valid_f1 = [f for f in f1_results if not np.isnan(f)]
    valid_cd = [c for c in cd_results if not np.isnan(c)]
    valid_iou = [i for i in iou_results if not np.isnan(i)]

    if valid_f1:
        print(f"F1 Score (mean): {np.mean(valid_f1):.4f}")
        print(f"F1 Score (median): {np.median(valid_f1):.4f}")
    else:
        print("F1 Score: No valid results.")
        
    if valid_cd:
        print(f"Chamfer Distance (mean, x1000): {np.mean(valid_cd) * 1000:.4f}")
        print(f"Chamfer Distance (median, x1000): {np.median(valid_cd) * 1000:.4f}")
    else:
        print("Chamfer Distance: No valid results.")

    if valid_iou:
        print(f"Volumetric IoU (mean): {np.mean(valid_iou):.4f}")
        print(f"Volumetric IoU (median): {np.median(valid_iou):.4f}")
    else:
        print("Volumetric IoU: No valid results.")

    if error_files_metric:
        print(f"\nEncountered errors or skips with {len(error_files_metric)} files during metric computation:")
        for f_err in error_files_metric[:5]:  # 只显示前5个错误
            print(f" - {f_err}")
        if len(error_files_metric) > 5:
            print(f" - ... and {len(error_files_metric) - 5} more")
            
    print(f"Successfully computed metrics for {len(valid_f1)} files (F1), {len(valid_cd)} files (CD), {len(valid_iou)} files (IoU) out of {len(common_files)} common files.")

def main():
    print("Starting Assembly Pipeline Evaluation Process...")
    
    # 设置要处理的样本数量
    MAX_SAMPLES = 10  # 可以根据需要调整
    
    # --- Step 0: 生成真值 STL 文件 ---
    gt_count = generate_ground_truth_stls_for_evaluation(max_samples=MAX_SAMPLES)
    if gt_count == 0:
        print("Error: No ground truth STL files were generated.")
        return

    # --- 加载模型 ---
    print("Loading Hugging Face model and tokenizer for assembly pipeline...")
    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
            model_max_length=1024
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load Hugging Face model/tokenizer from {HF_MODEL_PATH}. Exception: {e}")
        traceback.print_exc()
        return

    # --- Step 1: 生成预测 STL 文件 ---
    pred_count = generate_predicted_stls(model, tokenizer, max_samples=MAX_SAMPLES)

    # --- Step 2: 计算指标 ---
    compute_all_metrics()

    print(f"\nEvaluation process finished.")
    print(f"Ground truth STLs: {gt_count}, Predicted STLs: {pred_count}")

if __name__ == "__main__":
    main()