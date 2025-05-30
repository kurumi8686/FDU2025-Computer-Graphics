import os, re, json, torch, traceback, tempfile, subprocess, platform, random, glob
import cadquery as cq
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, List, Union, Dict, Any
import warnings
import requests
import time
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")

# --- SiliconFlow API Configuration ---
SILICONFLOW_API_KEY = "sk-fqhmppxgujazozedmuxzntmbhxmwkvxmkuzkvmqrcdtzzhtk"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# --- Hugging Face Model & Tokenizer ---
HF_MODEL_PATH = "/root/Workspace/Text-to-CadQuery/train/checkpoints_qwen3b/checkpoint-23292"
# --- Blender Configuration ---
BLENDER_EXECUTABLE_PATH = "/opt/blender/blender-3.2.2-linux-x64/blender"
BLENDER_SCRIPT_PATH = "blender_script.py"

# === SiliconFlow API Functions ===
def call_siliconflow_api(prompt: str, model: str = SILICONFLOW_MODEL) -> Optional[str]:
    """
    Call SiliconFlow API with the given prompt
    """
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.7,
        "max_tokens": 2048,
        "stream": False
    }
    
    max_retries = 3
    for retry_count in range(max_retries):
        try:
            print(f"调用SiliconFlow API，模型: {model}")
            response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=90)
            print(f"API响应状态码: {response.status_code}")
            
            # 处理速率限制错误
            if response.status_code == 429:
                error_msg = response.text
                print(f"遇到速率限制: {error_msg[:200]}...")
                wait_time = 30  # 默认等待10秒
                if "retry after" in error_msg.lower():
                    try:
                        wait_match = re.search(r"retry after (\d+)", error_msg.lower())
                        if wait_match:
                            wait_time = int(wait_match.group(1))
                    except:
                        pass
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                continue
            
            if response.status_code != 200:
                print(f"API错误响应: {response.text[:200]}...")
                response.raise_for_status()
                
            response_data = response.json()
            
            # 检查响应格式
            if not response_data or not isinstance(response_data, dict) or "choices" not in response_data:
                print(f"API响应格式错误: {str(response_data)[:100]}...")
                return None
                
            if len(response_data["choices"]) == 0:
                print("API返回的choices为空")
                return None
                
            choice = response_data["choices"][0]
            if not isinstance(choice, dict) or "message" not in choice:
                print("API返回的choice格式错误")
                return None
                
            content = choice["message"].get("content", "")
            if not isinstance(content, str):
                content = str(content)
                
            return content.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API请求错误: {str(e)}")
            if retry_count < max_retries - 1:
                wait_time = (retry_count + 1) * 5  # 递增等待时间
                print(f"等待 {wait_time} 秒后重试 ({retry_count+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数")
                return None
        except Exception as e:
            print(f"调用SiliconFlow API时出错: {e}")
            return None
    
    return None

def analyze_text_for_parts(text: str) -> Tuple[bool, List[str]]:
    """
    Analyze if the text describes multiple parts and decompose if needed
    """
    analysis_prompt = f"""
    Analyze the following text description and determine if it describes a 3D model with multiple distinct parts/components:

    Text: "{text}"

    Please answer in JSON format:
    {{
        "has_multiple_parts": true/false,
        "parts": ["part 1 description", "part 2 description", ...] or []
    }}

    Guidelines:
    - Only consider it multiple parts when there are clearly separate, distinct components
    - Simple modifications like holes, cuts, or surface features don't count as separate parts
    - Each part should be an independent geometric component
    - If has_multiple_parts is false, leave parts empty
    - If has_multiple_parts is true, break down the description into separate component descriptions

    Examples:
    - "A cube with a cylinder on top" → has_multiple_parts: true, parts: ["A cube", "A cylinder"]
    - "A cube with a hole" → has_multiple_parts: false, parts: []
    - "A table (with tabletop and four legs)" → has_multiple_parts: true, parts: ["Rectangular tabletop", "Four legs"]
    """
    
    response = call_siliconflow_api(analysis_prompt)
    if not response:
        return False, []
    
    try:
        # 提取JSON部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(response)
        
        has_multiple = result.get("has_multiple_parts", False)
        parts = result.get("parts", [])
        
        print(f"分析结果: has_multiple_parts={has_multiple}, parts={parts}")
        return has_multiple, parts
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response from analysis: {e}")
        print(f"Response was: {response}")
        return False, []

def integrate_cadquery_codes(codes: List[str], original_description: str) -> str:
    """
    Integrate multiple CadQuery codes into a single Assembly
    """
    # Join parts with newlines
    parts_str = "---\n"
    for i, code in enumerate(codes):
        parts_str += f"PART {i+1}:\n{code}\n\n"
        parts_str += "---\n"
    
    integration_prompt = f"""
    You are a CadQuery expert. I have multiple CadQuery code snippets that create individual parts, and I need you to integrate them into a CadQuery Assembly.

    Original description: "{original_description}"

    Individual CadQuery codes:
    {parts_str}

    Please create a single CadQuery script that:
    1. Creates each part defined in the individual code snippets
    2. Combines them into a CadQuery Assembly with APPROPRIATE positioning
    3. Ensures the final assembly matches the original description
    4. The last line should assign the Assembly to a variable (typically 'result' or 'assembly')

    Important rules:
    - Import cadquery as cq at the beginning
    - Use correct CadQuery Assembly syntax
    - Logically position parts according to the original description (like "on top", "beside", etc.)
    - Don't include any export statements
    - Don't include any show() calls
    - The final variable should contain the complete Assembly
    - Use REASONABLE position OFFSETS to ensure parts don't overlap

    Return only Python/CadQuery code, no markdown formatting.

    first Example format:

    import cadquery as cq
    # --- Part 1: Cylinder with Hollow Center ---
    part_1_outer_radius = 0.306 * 0.612  # Sketch radius scaled
    part_1_inner_radius = 0.1875 * 0.612  # Inner radius scaled
    part_1_height = 0.45
    part_1 = (
        cq.Workplane("XY")
        .circle(part_1_outer_radius)
        .extrude(part_1_height)
        .cut(cq.Workplane("XY").circle(part_1_inner_radius).extrude(part_1_height))
    )
    # --- Coordinate System Transformation for Part 1 ---
    part_1 = part_1.translate((0.069, 0.069, 0))
    # --- Part 2: Ring ---
    part_2_outer_radius = 0.375 * 0.75  # Sketch radius scaled
    part_2_inner_radius = 0.306 * 0.75  # Inner radius scaled
    part_2_height = 0.15
    part_2 = (
        cq.Workplane("XY")
        .circle(part_2_outer_radius)
        .extrude(part_2_height)
        .cut(cq.Workplane("XY").circle(part_2_inner_radius).extrude(part_2_height))
    )
    # --- Assembly ---
    assembly = part_1.union(part_2)

    ---
    second Example format:

    import cadquery as cq

    # --- Part 1: Cylinder ---
    part_1_radius = 0.035 * 0.07  # Sketch radius scaled
    part_1_height = 0.1875 + 0.1875

    part_1 = cq.Workplane("XY").circle(part_1_radius).extrude(part_1_height)

    # --- Coordinate System Transformation for Part 1 ---
    part_1 = part_1.rotate((0, 0, 0), (1, 0, 0), -90)
    part_1 = part_1.rotate((0, 0, 0), (0, 0, 1), -90)
    part_1 = part_1.translate((0.375, 0, 0))

    # --- Part 2: Cut Cylinder ---
    part_2_radius = 0.025 * 0.05  # Sketch radius scaled
    part_2_height = 0.05

    part_2 = cq.Workplane("XY").circle(part_2_radius).extrude(part_2_height)

    # --- Coordinate System Transformation for Part 2 ---
    part_2 = part_2.rotate((0, 0, 0), (1, 0, 0), -90)
    part_2 = part_2.rotate((0, 0, 0), (0, 0, 1), -90)
    part_2 = part_2.translate((0.75, 0.01, 0.01))

    # --- Part 3: Cut Cylinder ---
    part_3_radius = 0.025 * 0.05  # Sketch radius scaled
    part_3_height = 0.05

    part_3 = cq.Workplane("XY").circle(part_3_radius).extrude(part_3_height)

    # --- Coordinate System Transformation for Part 3 ---
    part_3 = part_3.rotate((0, 0, 0), (1, 0, 0), 90)
    part_3 = part_3.rotate((0, 0, 0), (0, 0, 1), -90)
    part_3 = part_3.translate((0, 0.06, 0.01))

    # --- Assembly ---
    assembly = part_1.cut(part_2).cut(part_3)

    """

    
    response = call_siliconflow_api(integration_prompt)
    print(f"Assembly集成响应:\n{response}")
    return response if response else ""

def fix_cadquery_code_with_ai(failed_code: str, error_message: str, original_description: str) -> str:
    """
    Use SiliconFlow API to fix CadQuery code based on error message
    """
    fix_prompt = f"""
    You are a CadQuery expert. The following CadQuery code failed to execute with an error. Please fix the code.

    Original description: "{original_description}"

    Failed code:
    ```python
    {failed_code}
    ```

    Error message:
    {error_message}

    Please provide a corrected version of the CadQuery code that:
    1. Fixes the specific error mentioned
    2. Still fulfills the original description
    3. Uses proper CadQuery syntax
    4. Imports cadquery as cq at the beginning
    5. Assigns the final result to a variable (typically 'result' or 'assembly')
    6. Does not include export statements or show() calls

    Return only the corrected Python/CadQuery code, no markdown formatting or explanations.

    Common fixes:
    - If Assembly error: Make sure to use cq.Assembly() and proper .add() syntax
    - If Shape error: Ensure all shapes are valid Workplane objects
    - If Location error: Use cq.Location(cq.Vector(x, y, z)) for positioning
    - If attribute error: Check method names and object types
    """
    
    response = call_siliconflow_api(fix_prompt)
    print(f"AI修复响应:\n{response}")
    return response if response else ""

# === Import functions from vis.py ===
def generate_cadquery_from_text(prompt_text: str, model, tokenizer) -> Optional[str]:
    """
    Uses a fine-tuned model to generate CadQuery code from a natural language prompt.
    """
    print(f"Generating CadQuery code for: {prompt_text[:100]}...")
    constraints = (
        "IMPORTANT RULES FOR CODE GENERATION:\n"
        "- The CadQuery code should ONLY define the requested object(s) using basic shapes like 'box', 'cylinder', etc.\n"
        "- Do NOT add any extra features, operations, or modifications like fillets, chamfers, or helper lines unless explicitly asked.\n"
        "- Do NOT introduce arbitrary scaling or multiplication for dimensions; use the numbers provided in the instruction directly.\n"
        "- Ensure the final line of code assigns the created CadQuery object (Workplane or Assembly) to a variable, typically 'result' or the last part name."
    )
    instruction_content = f"{prompt_text}\n\n{constraints}"
    full_prompt_for_model = f"### Instruction:\n{instruction_content}\n\n### Response:\n"
    inputs = tokenizer(full_prompt_for_model, return_tensors="pt", padding=False, truncation=True).to(model.device)
    input_lengths = inputs["input_ids"].shape[1]

    max_new_tokens = tokenizer.model_max_length - input_lengths - 15
    if max_new_tokens <= 500:
        print(f"Warning: Prompt is too long relative to model_max_length. Only {max_new_tokens} tokens left for generation.")

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        generated_ids = outputs[0, input_lengths:]
        response_part = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if not response_part:
            print("Error: Model generated empty code.")
            return None
        return response_part

    except Exception as e:
        print(f"Error during CadQuery code generation: {e}")
        traceback.print_exc()
        return None

def extract_and_clean_cadquery_code(text: str) -> str:
    """
    Extracts and cleans CadQuery Python code from model output.
    """
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0].strip()
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()

    text = text.replace("```python", "").replace("```", "").strip()
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if "from cadquery.vis import show" in line or "import cadquery.vis" in line:
            continue
        if "show_object(" in line or ".show(" in line:
            continue
        if re.search(r'cq\.exporters\.export\(.*?\)', line):
            continue
        cleaned_lines.append(line)
    return "\n".join(filter(None, cleaned_lines)).strip()

def run_cadquery_and_save_stl(code_str: str, output_stl_path: str) -> Tuple[bool, str]:
    """
    Executes CadQuery code and saves the result as an STL file.
    Also generates a GLB file from the STL.
    """
    print(f"Running CadQuery code and saving STL to: {output_stl_path}")
    
    #add glb file_path
    output_dir = os.path.dirname(output_stl_path)
    base_name = os.path.splitext(os.path.basename(output_stl_path))[0]
    # Ensure 'glb' directory exists in the same directory as the script or a predefined output path
    # For simplicity, let's assume 'glb' is a subdirectory in the current working directory or script's directory
    glb_output_dir = "glb" 
    os.makedirs(glb_output_dir, exist_ok=True)
    output_glb_path = os.path.join(glb_output_dir, f"{base_name}.glb")

    try:
        exec_globals = {'cq': cq}
        exec_locals = {}
        exec(code_str, exec_globals, exec_locals)
        result_obj = None
        found_assembly = None
        found_workplane = None

        for k, v in exec_locals.items():
            if isinstance(v, cq.Assembly):
                found_assembly = v
                break
            elif isinstance(v, cq.Workplane):
                found_workplane = v
        
        if found_assembly:
            result_obj = found_assembly
            print("Found CadQuery Assembly object.")
        elif found_workplane:
            result_obj = found_workplane
            print("Found CadQuery Workplane object.")

        if result_obj:
            if isinstance(result_obj, cq.Assembly):
                if result_obj.objects:
                    try:
                        # Attempt to create a compound from all valid shapes in the assembly
                        shapes_to_compound = [obj.val() for obj in result_obj.objects.values() if hasattr(obj, 'val') and isinstance(obj.val(), cq.Shape) and obj.val().isValid()]
                        if not shapes_to_compound:
                             return False, "Assembly contains no valid shapes to export."
                        compound = cq.Compound.makeCompound(shapes_to_compound)
                        if compound and compound.isValid():
                            cq.exporters.export(compound, output_stl_path)
                        else: # Fallback if compound creation failed or no valid shapes
                            print("Assembly found but could not create a valid compound. Trying to export the first valid object.")
                            # Try exporting the first valid shape if compound fails
                            exported_fallback = False
                            for obj_item in result_obj.objects.values():
                                if hasattr(obj_item, 'val') and isinstance(obj_item.val(), cq.Shape) and obj_item.val().isValid():
                                    cq.exporters.export(obj_item.val(), output_stl_path)
                                    exported_fallback = True
                                    break
                            if not exported_fallback:
                                return False, "Assembly object found but it's empty or no valid shape to export."
                    except Exception as assembly_export_e:
                        return False, f"Error exporting CadQuery Assembly: {assembly_export_e}\n{traceback.format_exc()}"
                else:
                    return False, "CadQuery Assembly object found, but it's empty."

            elif isinstance(result_obj, cq.Workplane):
                solid_to_export = result_obj.val()
                if not solid_to_export or not solid_to_export.isValid(): # Check if the solid is valid
                    return False, "CadQuery Workplane did not result in a valid solid."
                if isinstance(solid_to_export, cq.Compound) and not solid_to_export.Solids(): # Check for empty compound
                    return False, "CadQuery Workplane resulted in an empty compound."
                cq.exporters.export(solid_to_export, output_stl_path)

            print(f"Successfully generated STL: {output_stl_path}")

            #add generate glb
            print(f"Exporting GLB file to: {output_glb_path}")
            try:
                if not os.path.exists(output_stl_path) or os.path.getsize(output_stl_path) == 0:
                    return False, "STL file not created or is empty, cannot convert to GLB."

                # Check if assimp is available
                try:
                    subprocess.run(["assimp", "version"], check=True, capture_output=True, timeout=10)
                except FileNotFoundError:
                    return False, "assimp command not found. Please install Open Asset Import Library (assimp) and ensure it's in PATH."
                except subprocess.TimeoutExpired:
                    return False, "assimp command timed out. Check assimp installation."


                # Execute conversion
                conv_result = subprocess.run(
                    ["assimp", "export", output_stl_path, output_glb_path],
                    capture_output=True,
                    text=True,
                    timeout=60 
                )

                if conv_result.returncode != 0:
                    return False, f"GLB conversion failed with assimp: {conv_result.stderr}"

                if not os.path.exists(output_glb_path) or os.path.getsize(output_glb_path) == 0:
                    return False, "GLB file was not created or is empty after assimp conversion."

                print(f"Successfully generated GLB: {output_glb_path}")
                return True, f"Successfully generated STL: {output_stl_path} and GLB: {output_glb_path}"
            except Exception as glb_e:
                return False, f"Error exporting GLB: {glb_e}\n{traceback.format_exc()}"
        else:
            return False, "No CadQuery Workplane or Assembly object found in the executed code."

    except Exception as e:
        return False, f"Error executing CadQuery code or saving STL/GLB: {e}\n{traceback.format_exc()}"

def render_stl_with_blender(
        stl_file_path: str,
        output_render_dir: str,
        blender_executable: str,
        blender_script: str,
        render_timeout: int = 300,
        engine: str = "CYCLES"
) -> Optional[str]:
    """
    Renders an STL file to a PNG image using Blender.
    """
    print(f"Rendering STL file {stl_file_path} using Blender.")

    if not os.path.exists(stl_file_path):
        print(f"Error: STL file not found at {stl_file_path}")
        return None
    if not os.path.exists(blender_executable):
        print(f"Error: Blender executable not found at {blender_executable}")
        return None
    if not os.path.exists(blender_script):
        print(f"Error: Blender script not found at {blender_script}")
        return None

    os.makedirs(output_render_dir, exist_ok=True)

    blender_args = [
        "--object_path", stl_file_path,
        "--output_dir", output_render_dir,
        "--engine", engine,
        "--num_renders", "1"
    ]
    command = [
                  blender_executable,
                  "-b",
                  "--python", blender_script,
                  "--"
              ] + blender_args

    try:
        process = subprocess.run(
            command, timeout=render_timeout, check=False,
            capture_output=True, text=True
        )
        if process.returncode != 0:
            print(f"Blender process failed with return code {process.returncode}.")
            return None
    except subprocess.TimeoutExpired:
        print(f"Blender rendering timed out after {render_timeout} seconds.")
        return None
    except Exception as e:
        print(f"An error occurred while running Blender: {e}")
        traceback.print_exc()
        return None

    expected_image_path = os.path.join(output_render_dir, "render.png")
    if os.path.exists(expected_image_path):
        print(f"Successfully rendered image: {expected_image_path}")
        return expected_image_path
    else:
        png_files = glob.glob(os.path.join(output_render_dir, "*.png"))
        if png_files:
            return png_files[0]
        return None

# === Main Assembly Pipeline ===
def assembly_text_to_image_pipeline(
        user_prompt: str,
        hf_model,
        hf_tokenizer,
        final_output_image_path: str = "assembly_generated_image.png",
        max_fix_attempts: int = 3,
        eval_persistent_stl_path: Optional[str] = None  # 新增参数，用于评估时保存STL
):
    """
    Enhanced pipeline that handles multi-part assemblies using SiliconFlow API
    """
    print(f"=== Starting Assembly Pipeline ===")
    print(f"Input prompt: {user_prompt}")
    
    # Step 1: Analyze if the text describes multiple parts using SiliconFlow
    print("\nStep 1: Analyzing text for multiple parts using SiliconFlow...")
    has_multiple_parts, parts = analyze_text_for_parts(user_prompt)
    
    if has_multiple_parts and parts:
        print(f"Found {len(parts)} parts:")
        for i, part in enumerate(parts):
            print(f"  Part {i+1}: {part}")
        
        # Step 2: Generate CadQuery code for each part using pretrained model
        print("\nStep 2: Generating CadQuery code for each part...")
        part_codes = []
        for i, part_description in enumerate(parts):
            print(f"\nGenerating code for part {i+1}...")
            cadquery_code = generate_cadquery_from_text(part_description, hf_model, hf_tokenizer)
            if cadquery_code:
                cleaned_code = extract_and_clean_cadquery_code(cadquery_code)
                if cleaned_code:
                    part_codes.append(cleaned_code)
                    print(f"Part {i+1} code generated successfully")
                    print(f"Part {i+1} code:\n---\n{cleaned_code}\n---")
                else:
                    print(f"Warning: Part {i+1} code is empty after cleaning")
            else:
                print(f"Warning: Failed to generate code for part {i+1}")
        
        if not part_codes:
            print("Error: No valid part codes generated. Falling back to single-part generation.")
            has_multiple_parts = False
        else:
            # Step 3: Integrate all parts into a single Assembly using SiliconFlow
            print(f"\nStep 3: Integrating {len(part_codes)} parts into Assembly using SiliconFlow...")
            final_cadquery_code = integrate_cadquery_codes(part_codes, user_prompt)
            if not final_cadquery_code:
                print("Error: Failed to integrate parts. Falling back to single-part generation.")
                has_multiple_parts = False
    
    # Fallback to single-part generation
    if not has_multiple_parts:
        print("\nGenerating as single part...")
        cadquery_code = generate_cadquery_from_text(user_prompt, hf_model, hf_tokenizer)
        if not cadquery_code:
            print("Pipeline aborted: Failed to generate CadQuery code.")
            return None, None # 返回 None, None 表示失败
        final_cadquery_code = extract_and_clean_cadquery_code(cadquery_code)
        if not final_cadquery_code:
            print("Pipeline aborted: Failed to clean CadQuery code.")
            return None, None # 返回 None, None 表示失败

    print(f"\nFinal CadQuery Code:\n---\n{final_cadquery_code}\n---")

    # Rest of the pipeline with STL generation retry mechanism
    actual_stl_path_for_run: Optional[str] = None
    temp_dir_obj = None # 用于管理临时目录对象

    if eval_persistent_stl_path:
        eval_stl_dir = os.path.dirname(eval_persistent_stl_path)
        os.makedirs(eval_stl_dir, exist_ok=True)
        actual_stl_path_for_run = eval_persistent_stl_path
        
        # 为Blender输出创建一个基于STL文件名的子目录，以保持整洁
        stl_basename_no_ext = os.path.splitext(os.path.basename(eval_persistent_stl_path))[0]
        blender_output_subdir_base = os.path.join(eval_stl_dir, "blender_render_temp")
        blender_output_subdir = os.path.join(blender_output_subdir_base, stl_basename_no_ext)
        os.makedirs(blender_output_subdir, exist_ok=True)
    else:
        temp_dir_obj = tempfile.TemporaryDirectory()
        temp_dir_path = temp_dir_obj.name
        print(f"Using temporary directory: {temp_dir_path}")
        temp_stl_filename = "model.stl"
        actual_stl_path_for_run = os.path.join(temp_dir_path, temp_stl_filename)
        blender_output_subdir = os.path.join(temp_dir_path, "blender_render_output")
        os.makedirs(blender_output_subdir, exist_ok=True)

    # 尝试生成STL，如果失败则使用AI修复代码
    current_code = final_cadquery_code
    stl_success = False
    stl_message = ""
    
    for attempt in range(max_fix_attempts + 1):  # 原始尝试 + 修复尝试
        print(f"\n--- STL生成尝试 {attempt + 1}/{max_fix_attempts + 1} ---")
        
        # Generate STL from CadQuery
        stl_success, stl_message = run_cadquery_and_save_stl(current_code, actual_stl_path_for_run)
        
        if stl_success:
            print(f"STL生成成功！路径: {actual_stl_path_for_run}")
            break
        else:
            print(f"STL生成失败: {stl_message}")
            if attempt < max_fix_attempts:
                print(f"\n尝试使用AI修复代码...")
                fixed_code = fix_cadquery_code_with_ai(
                    failed_code=current_code,
                    error_message=stl_message,
                    original_description=user_prompt
                )
                if fixed_code and fixed_code.strip():
                    current_code = extract_and_clean_cadquery_code(fixed_code)
                    print(f"\nAI修复后的代码:\n---\n{current_code}\n---")
                else:
                    print("AI修复失败，无法生成有效代码")
                    break
            else:
                print(f"已达到最大修复尝试次数 ({max_fix_attempts})，放弃修复")
                break

    if not stl_success:
        print(f"Pipeline aborted: Failed to generate STL after {max_fix_attempts + 1} attempts.")
        print(f"Final error: {stl_message}")
        if temp_dir_obj: # 清理临时目录
            temp_dir_obj.cleanup()
        return None, None # 返回 None, None 表示失败

    if not os.path.exists(actual_stl_path_for_run) or os.path.getsize(actual_stl_path_for_run) == 0:
        print(f"Pipeline aborted: STL file was not created or is empty at {actual_stl_path_for_run}")
        if temp_dir_obj: # 清理临时目录
            temp_dir_obj.cleanup()
        return None, actual_stl_path_for_run # 返回 None 和预期的STL路径

    # Render STL to Image
    rendered_image_temp_path = render_stl_with_blender(
        stl_file_path=actual_stl_path_for_run,
        output_render_dir=blender_output_subdir,
        blender_executable=BLENDER_EXECUTABLE_PATH,
        blender_script=BLENDER_SCRIPT_PATH
    )

    final_image_result_path = None
    if rendered_image_temp_path:
        try:
            final_output_dir = os.path.dirname(final_output_image_path)
            if final_output_dir:
                os.makedirs(final_output_dir, exist_ok=True)
            import shutil
            shutil.copy(rendered_image_temp_path, final_output_image_path)
            print(f"Final image saved to: {final_output_image_path}")
            final_image_result_path = final_output_image_path
        except Exception as e:
            print(f"Error copying rendered image to final path: {e}")
            final_image_result_path = rendered_image_temp_path # 返回临时路径
    else:
        print("Pipeline: Failed to render STL to image.")

    if temp_dir_obj: # 清理临时目录
        temp_dir_obj.cleanup()
        
    return final_image_result_path, actual_stl_path_for_run if stl_success else None


if __name__ == "__main__":
    # Load model and tokenizer
    print("Loading Hugging Face model and tokenizer...")
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
    except Exception as e:
        print(f"Fatal Error: Could not load Hugging Face model/tokenizer from {HF_MODEL_PATH}. Exception: {e}")
        traceback.print_exc()
        exit()

    # Check Blender paths
    if not os.path.exists(BLENDER_EXECUTABLE_PATH):
        print(f"FATAL ERROR: Blender executable not found at '{BLENDER_EXECUTABLE_PATH}'.")
        exit()
    if not os.path.exists(BLENDER_SCRIPT_PATH):
        print(f"FATAL ERROR: Blender script '{BLENDER_SCRIPT_PATH}' not found.")
        exit()

    # Get user input
    user_text_prompt = input("Enter your text prompt to generate a 3D model assembly: ")
    output_file = "assembly_generated_image.png"

    print(f"\n--- Starting Assembly Text-to-Image Generation ---")
    # 在直接运行时，不传递 eval_persistent_stl_path
    final_image, generated_stl = assembly_text_to_image_pipeline(
        user_text_prompt, model, tokenizer, 
        final_output_image_path=output_file,
        max_fix_attempts=3
    )
    
    if final_image:
        print(f"\n--- Assembly Pipeline Succeeded ---")
        print(f"Generated image saved at: {os.path.abspath(final_image)}")
        if generated_stl:
            print(f"Generated STL (transient or for eval) at: {os.path.abspath(generated_stl)}")
    else:
        print(f"\n--- Assembly Pipeline Failed ---")
        print("Please check the logs above for errors.")