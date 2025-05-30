import glob
import json
import multiprocessing
import os
import platform
import random
import subprocess
import tempfile
import time
import zipfile
from tqdm import tqdm
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union
import fire
import fsspec
import GPUtil
import pandas as pd
from loguru import logger
import objaverse.xl as oxl
from objaverse.utils import get_uid_from_str


def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.
    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.
    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"~/.objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")


def zipdir(path: str, ziph: zipfile.ZipFile) -> None:
    """Zip up a directory with an arcname structure.
    Args:
        path (str): Path to the directory to zip.
        ziph (zipfile.ZipFile): ZipFile handler object to write to.

    Returns:
        None
    """
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            # this ensures the structure inside the zip starts at folder/
            arcname = os.path.join(os.path.basename(root), file)
            ziph.write(os.path.join(root, file), arcname=arcname)


def handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """Process a single 3D object, rendering it as a single PNG image.

    Args:
        local_path: Path to the 3D object file
        file_identifier: Identifier for the file
        sha256: SHA256 hash of the file
        metadata: Additional metadata
        num_renders: Number of renders (kept for compatibility)
        render_dir: Output directory for renders
        only_northern_hemisphere: Whether to only render the northern hemisphere
        gpu_devices: GPU device configuration
        render_timeout: Timeout in seconds for the rendering process
        successful_log_file: File to log successful renders
        failed_log_file: File to log failed renders

    Returns:
        bool: True if rendering was successful, False otherwise
    """
    # Use file identifier as the base for output filename
    save_uid = file_identifier.split('.')[0]
    args = f"--object_path '{local_path}' --num_renders {num_renders}"

    # Handle GPU selection
    using_gpu: bool = True
    gpu_i = 0
    if isinstance(gpu_devices, int) and gpu_devices > 0:
        num_gpus = gpu_devices
        gpu_i = random.randint(0, num_gpus - 1)
    elif isinstance(gpu_devices, list):
        gpu_i = random.choice(gpu_devices)
    elif isinstance(gpu_devices, int) and gpu_devices == 0:
        using_gpu = False
    else:
        raise ValueError(
            f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
        )

    # Create output directory
    os.makedirs(render_dir, exist_ok=True)
    target_directory = os.path.join(render_dir, save_uid)
    os.makedirs(target_directory, exist_ok=True)
    args += f" --output_dir {target_directory}"

    # Check system type and set engine
    if platform.system() == "Linux" and using_gpu:
        args += " --engine CYCLES"
    elif platform.system() == "Darwin" or (platform.system() == "Linux" and not using_gpu):
        args += " --engine CYCLES"
    else:
        raise NotImplementedError(f"Platform {platform.system()} is not supported.")

    # Set hemisphere rendering option
    if only_northern_hemisphere:
        args += " --only_northern_hemisphere"

    # Build command 这是你在Linux上的Blender实际路径
    BLENDER_EXECUTABLE_PATH = "/opt/blender/blender-3.2.2-linux-x64/blender"
    command = f"{BLENDER_EXECUTABLE_PATH} -b --python blender_script.py -- {args}"

    # Execute rendering command
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            timeout=render_timeout,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print("STDOUT:", result.stdout.decode())
        print("STDERR:", result.stderr.decode())

    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after {render_timeout} seconds")
        print("STDOUT:", e.stdout.decode() if e.stdout else "")
        print("STDERR:", e.stderr.decode() if e.stderr else "")
        if failed_log_file is not None:
            log_processed_object(
                failed_log_file,
                file_identifier,
                sha256,
            )
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        if failed_log_file is not None:
            log_processed_object(
                failed_log_file,
                file_identifier,
                sha256,
            )
        return False

    # Check if PNG file was successfully rendered
    png_files = glob.glob(os.path.join(target_directory, "*.png"))
    if len(png_files) == 0:
        logger.error(f"Found object {file_identifier} was not rendered successfully!")
        if failed_log_file is not None:
            log_processed_object(
                failed_log_file,
                file_identifier,
                sha256,
            )
        return False

    # Log success
    if successful_log_file is not None:
        log_processed_object(successful_log_file, file_identifier, sha256)

    return True


def handle_new_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-new-object.csv",
) -> None:
    """Called when a new object is found.

    Here, the object is not used in Objaverse-XL, but is still downloaded with the
    repository. The object may have not been used because it does not successfully
    import into Blender. If None, the object will be downloaded, but nothing will be
    done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): The file identifier of the new 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, including the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save the handle_new_object logs to.

    Returns:
        None
    """
    # log the new object
    log_processed_object(log_file, file_identifier, sha256)


def handle_modified_object(
    local_path: str,
    file_identifier: str,
    new_sha256: str,
    old_sha256: str,
    metadata: Dict[str, Any],
    num_renders: int,
    render_dir: str,
    only_northern_hemisphere: bool,
    gpu_devices: Union[int, List[int]],
    render_timeout: int,
) -> None:
    """Called when a modified object is found and downloaded.

    Here, the object is successfully downloaded, but it has a different sha256 than the
    one that was downloaded with Objaverse-XL. This is not expected to happen very
    often, because the same commit hash is used for each repo. If None, the object will
    be downloaded, but nothing will be done with it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        new_sha256 (str): SHA256 of the contents of the newly downloaded 3D object.
        old_sha256 (str): Expected SHA256 of the contents of the 3D object as it was
            when it was downloaded with Objaverse-XL.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.

    Returns:
        None
    """
    success = handle_found_object(
        local_path=local_path,
        file_identifier=file_identifier,
        sha256=new_sha256,
        metadata=metadata,
        num_renders=num_renders,
        render_dir=render_dir,
        only_northern_hemisphere=only_northern_hemisphere,
        gpu_devices=gpu_devices,
        render_timeout=render_timeout,
        successful_log_file=None,
        failed_log_file=None,
    )

    if success:
        log_processed_object(
            "handle-modified-object-successful.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )
    else:
        log_processed_object(
            "handle-modified-object-failed.csv",
            file_identifier,
            old_sha256,
            new_sha256,
        )


def handle_missing_object(
    file_identifier: str,
    sha256: str,
    metadata: Dict[str, Any],
    log_file: str = "handle-missing-object.csv",
) -> None:
    """Called when an object that is in Objaverse-XL is not found.

    Here, it is likely that the repository was deleted or renamed. If None, nothing
    will be done with the missing object.

    Args:
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the original 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        log_file (str): Name of the log file to save missing renders to.

    Returns:
        None
    """
    # log the missing object
    log_processed_object(log_file, file_identifier, sha256)


def get_example_objects() -> pd.DataFrame:
    """Returns a DataFrame of example objects to use for debugging."""
    return pd.read_json("./file_list.json", orient="records")
    

def render_objects(
    render_dir: str = "../rendered_images_peft",  # 根目录下的 rendered_images 文件夹
    data_dir: str = "../generated_stl_peft",  # STL 文件所在的目录
    only_northern_hemisphere: bool = False,
    render_timeout: int = 300,
    gpu_devices: Optional[Union[int, List[int]]] = None,  # 保持 None 让它自动检测 GPU，或者设置为 0 使用 CPU
) -> None:
    """Renders 3D objects as single PNG images.

    Args:
        render_dir (str): Directory where the rendered images will be saved.
        data_dir (str): Directory containing the 3D model files.
        only_northern_hemisphere (bool): Whether to only render the northern hemisphere.
        render_timeout (int): Render timeout in seconds.
        gpu_devices: GPU device configuration.

    Returns:
        None
    """
    if platform.system() not in ["Linux", "Darwin"]:
        raise NotImplementedError(
            f"Platform {platform.system()} is not supported. Use Linux or MacOS."
        )

    # Get the GPU devices to use
    parsed_gpu_devices: Union[int, List[int]] = 0
    if gpu_devices is None:
        parsed_gpu_devices = len(GPUtil.getGPUs())
    logger.info(f"Using {parsed_gpu_devices} GPU devices for rendering.")

    # Scan the data directory for 3D model files
    model_files = []
    supported_extensions = ['.obj', '.stl', '.ply', '.glb', '.gltf', '.fbx', '.blend', '.usd', '.usda', '.dae', '.abc']
    
    for ext in supported_extensions:
        found_files = glob.glob(os.path.join(data_dir, f"*{ext}"))
        model_files.extend(found_files)
    
    logger.info(f"Found {len(model_files)} 3D model files in {data_dir}")
    
    if len(model_files) == 0:
        logger.error(f"No supported 3D model files found in {data_dir}")
        return

    # Create output directory
    os.makedirs(render_dir, exist_ok=True)
    
    # Check already rendered objects
    rendered_files = set()
    for filename in os.listdir(render_dir):
        if filename.endswith('.png'):
            # Get file name prefix (without extension)
            rendered_files.add(filename.split('.')[0])
    
    logger.info(f"Found {len(rendered_files)} already rendered objects")
    
    # Filter out the already rendered objects
    models_to_render = []
    for model_file in model_files:
        file_basename = os.path.basename(model_file)
        save_uid = file_basename.split('.')[0]
        if save_uid not in rendered_files:
            models_to_render.append(model_file)
    
    logger.info(f"Rendering {len(models_to_render)} new objects")
    
    # Sort the model files by name
    models_to_render.sort(key=lambda x: os.path.basename(x))
    
    # Render each model
    total_renders = 1
    for model_file in tqdm(models_to_render):
        file_basename = os.path.basename(model_file)
        handle_found_object(
            local_path=model_file,
            file_identifier=file_basename,
            sha256="unknown",  # Not tracking hashes
            metadata=None,
            num_renders=total_renders,
            render_dir=render_dir,
            only_northern_hemisphere=only_northern_hemisphere,
            gpu_devices=parsed_gpu_devices,
            render_timeout=render_timeout
        )

if __name__ == "__main__":
    fire.Fire(render_objects)
