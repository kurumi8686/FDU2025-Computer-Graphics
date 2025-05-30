import trimesh
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import os
from tqdm import tqdm

def sample_mesh(stl_path, num_points=10000):
    mesh = trimesh.load(stl_path)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def sample_mesh_normalized(stl_path, num_points=10000):
    mesh = trimesh.load(stl_path)
    mesh.apply_translation(-mesh.centroid)
    scale = np.max(mesh.bounding_box.extents)
    mesh.apply_scale(1.0 / scale)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def chamfer_distance(points1, points2):
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    chamfer = np.mean(dist1**2) + np.mean(dist2**2)
    return chamfer

def f1_score(points_pred, points_gt, threshold=0.02):
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
    """
    Normalize the mesh to fit inside a unit cube [0, 1]^3
    """
    mesh = mesh.copy()
    bounds = mesh.bounds
    scale = bounds[1] - bounds[0]
    max_extent = np.max(scale)
    mesh.apply_translation(-bounds[0])  # move to origin
    mesh.apply_scale(1.0 / max_extent)  # scale to fit into [0,1]
    return mesh

def voxelize_mesh(mesh, voxel_size=0.02):
    voxelized = mesh.voxelized(pitch=voxel_size)
    return voxelized.matrix

def volumetric_iou(mesh1, mesh2, voxel_size=0.02):
    # Normalize both meshes to [0,1]^3
    mesh1 = normalize_mesh(mesh1)
    mesh2 = normalize_mesh(mesh2)
    # Voxelize
    vox1 = voxelize_mesh(mesh1, voxel_size)
    vox2 = voxelize_mesh(mesh2, voxel_size)
    # Pad to same shape
    shape = np.maximum(vox1.shape, vox2.shape)
    vox1_padded = np.zeros(shape, dtype=bool)
    vox2_padded = np.zeros(shape, dtype=bool)
    vox1_padded[:vox1.shape[0], :vox1.shape[1], :vox1.shape[2]] = vox1
    vox2_padded[:vox2.shape[0], :vox2.shape[1], :vox2.shape[2]] = vox2
    # Compute IOU
    intersection = np.logical_and(vox1_padded, vox2_padded).sum()
    union = np.logical_or(vox1_padded, vox2_padded).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    return iou

'''
candidate_list = []
candidate_dir = "./eval_result"

for candidate in os.listdir(candidate_dir):
    if candidate.startswith("gemma"):
        candidate_path = os.path.join(candidate_dir, candidate)
        with open(candidate_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.endswith("Match: Yes"):
                    number = line.split(":")[0]
                    candidate_list.append(number)
'''


inference_path = '../generated_stl_peft'
gt_path = '../output_gt'

cd_results = []
error_files = []
f1_results = []
iou_results = []
processed_files_count = 0

# 1. Get lists of STL files from both directories
try:
    inference_files = {f for f in os.listdir(inference_path) if f.endswith('.stl') and not f.startswith('._')}
    gt_files = {f for f in os.listdir(gt_path) if f.endswith('.stl') and not f.startswith('._')}
except FileNotFoundError as e:
    print(f"Error: Directory not found - {e}. Please check paths.")
    exit()


# 2. Find common files
common_files = sorted(list(inference_files.intersection(gt_files)))

if not common_files:
    print("No common STL files found in both directories.")
else:
    # Limit to first 100 files
    common_files = common_files[:100]
    print(f"Processing first {len(common_files)} common STL files...")
    
    for filename in tqdm(common_files, desc="Comparing models"):
        # Construct full paths
        ground_truth_file_path = os.path.join(gt_path, filename)
        prediction_file_path = os.path.join(inference_path, filename)
        try:
            # Load meshes
            mesh_gt = trimesh.load(ground_truth_file_path, force='mesh') # force='mesh' to ensure mesh object
            mesh_pred = trimesh.load(prediction_file_path, force='mesh')

            # Check if meshes are valid (e.g., have vertices and faces)
            if not (mesh_gt.vertices.shape[0] > 0 and mesh_gt.faces.shape[0] > 0):
                print(f"Warning: Ground truth mesh {filename} is empty or invalid. Skipping.")
                error_files.append(filename + " (empty/invalid GT mesh)")
                continue
            if not (mesh_pred.vertices.shape[0] > 0 and mesh_pred.faces.shape[0] > 0):
                print(f"Warning: Prediction mesh {filename} is empty or invalid. Skipping.")
                error_files.append(filename + " (empty/invalid Predicted mesh)")
                continue

            points_gt = sample_mesh_normalized(ground_truth_file_path)
            points_pred = sample_mesh_normalized(prediction_file_path)

            # Calculate metrics
            iou = volumetric_iou(mesh_gt, mesh_pred, voxel_size=0.02) # Pass loaded mesh objects
            iou_results.append(iou)

            f1 = f1_score(points_pred, points_gt)
            f1_results.append(f1)
            cd = chamfer_distance(points_gt, points_pred)
            cd_results.append(cd)

            processed_files_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            error_files.append(filename + f" (error: {e})")

print("F1 mean: ",np.mean(f1_results))
print("F1 median: ",np.median(f1_results))
print("CD mean: ",np.mean(cd_results)*1000)
print("CD median: ",np.median(cd_results)*1000)
print("IOU mean: ",np.mean(iou_results))
print("IOU median: ",np.median(iou_results))
