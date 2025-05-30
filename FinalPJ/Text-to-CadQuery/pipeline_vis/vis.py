import os, re, json, torch, traceback, tempfile, subprocess, platform, random, glob
import cadquery as cq
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, List, Union, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.generation.configuration_utils")


# --- Hugging Face Model & Tokenizer ---
HF_MODEL_PATH = "../train/checkpoints_qwen3b/checkpoint-23292"
# --- Blender Configuration ---
BLENDER_EXECUTABLE_PATH = "/opt/blender/blender-3.2.2-linux-x64/blender"
BLENDER_SCRIPT_PATH = "blender_script.py"

# --- GPU Configuration for Rendering (from custom_main.py) ---
# Simplified, or you can integrate GPUtil if needed for more dynamic selection.
# For this script, Blender will handle GPU selection based on its settings
# or what the blender_script.py tries to configure.

# === STEP 1: Generate CadQuery Code from Text ===
def generate_cadquery_from_text(prompt_text: str, model, tokenizer) -> Optional[str]:
    """
    Uses a fine-tuned model to generate CadQuery code from a natural language prompt.
    """
    print(f"Step 1: Generating CadQuery code for input text...\n")
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
    print(f"Tokenizer model_max_length: {tokenizer.model_max_length}")
    print(f"Input tokens length (input_lengths): {input_lengths}")

    max_new_tokens = tokenizer.model_max_length - input_lengths - 15
    if max_new_tokens <= 100:
        print(f"Warning: Prompt is too long relative to model_max_length. Only {max_new_tokens} tokens left for generation.")
    print(f"Calculated max_new_tokens: {max_new_tokens}")

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,  # 通常eos_token_id作为pad_token_id
                do_sample=False  # 保持确定性输出
            )

        # 解码时，我们只关心新生成的部分
        generated_ids = outputs[0, input_lengths:]
        response_part = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"Generated CadQuery Code (raw response part):\n---\n{response_part}\n---")

        if not response_part:
            print("Error: Model generated empty code.")
            return None
        return response_part

    except Exception as e:
        print(f"Error during CadQuery code generation: {e}")
        traceback.print_exc()
        return None

# === STEP 2: Clean, Run CadQuery, and Generate STL ===
def extract_and_clean_cadquery_code(text: str) -> str:
    """
    Extracts and cleans CadQuery Python code from model output.
    Removes <|endoftext|>, markdown, and unnecessary show/export calls.
    """
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0].strip()
    if "### Response:" in text:  # Though generate_cadquery_from_text should handle this
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
    """
    print(f"Step 2: Running CadQuery code and saving STL to: {output_stl_path}")
    
    #add glb file_path
    output_dir = os.path.dirname(output_stl_path)
    base_name = os.path.splitext(os.path.basename(output_stl_path))[0]
    # output_glb_path = os.path.join(output_dir, f"{base_name}.glb")
    output_glb_path = os.path.join('glb', f"{base_name}.glb")


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
                break  # Prefer assembly
            elif isinstance(v, cq.Workplane):
                found_workplane = v
        if found_assembly:
            result_obj = found_assembly
            print("Found CadQuery Assembly object.")
        elif found_workplane:
            result_obj = found_workplane
            print("Found CadQuery Workplane object.")
        
        if not result_obj:
            return False, "No CadQuery Workplane or Assembly object found in the executed code."


        if result_obj:
            # For Assembly, export each solid if needed or the combined assembly
            if isinstance(result_obj, cq.Assembly):
                # Combine into a single compound for export if possible, or export the first shape
                # This might need more sophisticated handling based on your assembly structure
                if result_obj.objects:
                    try:
                        compound = cq.Compound.makeCompound([obj.val() for obj in result_obj.objects.values() if hasattr(obj, 'val') and isinstance(obj.val(), cq.Shape)])
                        if compound:
                            cq.exporters.export(compound, output_stl_path)
                        else:  # Fallback if compound creation failed or no valid shapes
                            print("Assembly found but could not create a compound. Trying to export the first object.")
                            first_obj_val = next(iter(result_obj.objects.values())).val()
                            if first_obj_val:
                                cq.exporters.export(first_obj_val, output_stl_path)
                            else:
                                return False, "Assembly object found but it's empty or no valid shape to export."

                    except Exception as assembly_export_e:
                        return False, f"Error exporting CadQuery Assembly: {assembly_export_e}\n{traceback.format_exc()}"
                else:
                    return False, "CadQuery Assembly object found, but it's empty."

            elif isinstance(result_obj, cq.Workplane):
                solid_to_export = result_obj.val()
                if isinstance(solid_to_export, cq.Compound) and not solid_to_export.Solids():
                    return False, "CadQuery Workplane resulted in an empty compound."
                if not solid_to_export:
                    return False, "CadQuery Workplane did not result in a valid solid."
                cq.exporters.export(solid_to_export, output_stl_path)
            
            print(f"Successfully generated STL: {output_stl_path}")

            #add generate glb
            print(f"Exporting GLB file to: {output_glb_path}")
            try:
                # 使用assimp进行转换
                if not os.path.exists(output_stl_path):
                    return False, "STL file not created for GLB conversion"

                # 检查assimp是否可用
                try:
                    subprocess.run(["assimp", "version"], check=True, capture_output=True)
                except FileNotFoundError:
                    return False, "assimp command not found. Please install Open Asset Import Library (assimp)"

                # 执行转换
                conv_result = subprocess.run(
                    ["assimp", "export", output_stl_path, output_glb_path],
                    capture_output=True,
                    text=True
                )

                if conv_result.returncode != 0:
                    return False, f"GLB conversion failed: {conv_result.stderr}"

                if not os.path.exists(output_glb_path):
                    return False, "GLB file was not created"

                print(f"Successfully generated GLB: {output_glb_path} and GLB: {output_glb_path}")
                return True, f"Successfully generated STL: {output_stl_path} and GLB: {output_glb_path}"
            except Exception as glb_e:
                return False, f"Error exporting GLB: {glb_e}\n{traceback.format_exc()}"


            # return True, f"Successfully generated STL: {output_stl_path}"
        # else:
           #  return False, "No CadQuery Workplane or Assembly object found in the executed code."

    except Exception as e:
        return False, f"Error executing CadQuery code or saving STL: {e}\n{traceback.format_exc()}"


# === STEP 3: Render STL to Image using Blender ===
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
    The blender_script.py is expected to save an image named "render.png"
    inside the 'output_render_dir'.
    """
    print(f"Step 3: Rendering STL file {stl_file_path} using Blender.")
    print(f"Blender output will be in: {output_render_dir}")

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

    # Arguments for blender_script.py
    blender_args = [
        "--object_path", stl_file_path,
        "--output_dir", output_render_dir,  # This is where blender_script.py will save "render.png"
        "--engine", engine,
        "--num_renders", "1"
    ]
    command = [
                  blender_executable,
                  "-b",
                  "--python", blender_script,
                  "--"
              ] + blender_args
    print(f"Executing Blender command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command, timeout=render_timeout, check=False,
            capture_output=True, text=True
        )
        print("Blender STDOUT:")
        print(process.stdout)
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

    # blender_script.py is expected to save "render.png" in the output_render_dir
    expected_image_path = os.path.join(output_render_dir, "render.png")
    if os.path.exists(expected_image_path):
        print(f"Successfully rendered image: {expected_image_path}")
        return expected_image_path
    else:
        print(f"Error: Rendered image not found at {expected_image_path}. Check Blender logs.")
        # Look for any PNG files if the name is different
        png_files = glob.glob(os.path.join(output_render_dir, "*.png"))
        if png_files:
            print(f"Found other PNG files: {png_files}. Using the first one: {png_files[0]}")
            return png_files[0]
        return None


# === Main Workflow Orchestration ===
def text_to_image_pipeline(
        user_prompt: str,
        hf_model,
        hf_tokenizer,
        final_output_image_path: str = "final_rendered_image.png"
):
    """
    Full pipeline from text prompt to a rendered image.
    """
    # Step 1: Generate CadQuery Code
    cadquery_code = generate_cadquery_from_text(user_prompt, hf_model, hf_tokenizer)
    if not cadquery_code:
        print("Pipeline aborted: Failed to generate CadQuery code.")
        return None

    cleaned_cadquery_code = extract_and_clean_cadquery_code(cadquery_code)
    if not cleaned_cadquery_code:
        print("Pipeline aborted: Failed to clean CadQuery code (empty after cleaning).")
        return None
    print(f"Cleaned CadQuery Code:\n---\n{cleaned_cadquery_code}\n---")

    # Create a temporary directory to store intermediate STL and render output
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        temp_stl_filename = "model.stl"  # Keep filename simple
        temp_stl_path = os.path.join(temp_dir, temp_stl_filename)

        # This will be the directory passed to Blender's --output_dir argument.
        # Blender will create "render.png" inside this.
        blender_output_subdir = os.path.join(temp_dir, "blender_render_output")
        os.makedirs(blender_output_subdir, exist_ok=True)

        # Step 2: Generate STL from CadQuery
        stl_success, stl_message = run_cadquery_and_save_stl(cleaned_cadquery_code, temp_stl_path)
        if not stl_success:
            print(f"Pipeline aborted: Failed to generate STL. Reason: {stl_message}")
            return None

        if not os.path.exists(temp_stl_path) or os.path.getsize(temp_stl_path) == 0:
            print(f"Pipeline aborted: STL file was not created or is empty at {temp_stl_path}")
            return None
        print(f"STL file generated at {temp_stl_path}, size: {os.path.getsize(temp_stl_path)} bytes.")

        # Step 3: Render STL to Image
        rendered_image_temp_path = render_stl_with_blender(
            stl_file_path=temp_stl_path,
            output_render_dir=blender_output_subdir,  # Blender saves "render.png" here
            blender_executable=BLENDER_EXECUTABLE_PATH,
            blender_script=BLENDER_SCRIPT_PATH
        )

        if not rendered_image_temp_path:
            print("Pipeline aborted: Failed to render STL to image.")
            return None

        # Copy the rendered image from the temporary Blender output dir to the final desired path
        try:
            final_output_dir = os.path.dirname(final_output_image_path)
            if final_output_dir:
                os.makedirs(final_output_dir, exist_ok=True)
            import shutil
            shutil.copy(rendered_image_temp_path, final_output_image_path)
            print(f"Final image saved to: {final_output_image_path}")
            return final_output_image_path
        except Exception as e:
            print(f"Error copying rendered image to final path: {e}")
            print(f"Rendered image is available at temporary location: {rendered_image_temp_path}")
            return rendered_image_temp_path


if __name__ == "__main__":
    # --- Load Hugging Face Model and Tokenizer ONCE ---
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
        print("Please ensure the HF_MODEL_PATH is correct and the model files are accessible.")
        traceback.print_exc()
        exit()

    # --- Check Blender Path ---
    if not os.path.exists(BLENDER_EXECUTABLE_PATH):
        print(f"FATAL ERROR: Blender executable not found at '{BLENDER_EXECUTABLE_PATH}'.")
        print("Please update the BLENDER_EXECUTABLE_PATH variable in this script.")
        exit()
    if not os.path.exists(BLENDER_SCRIPT_PATH):
        print(f"FATAL ERROR: Blender script '{BLENDER_SCRIPT_PATH}' not found.")
        print("Please ensure 'blender_script.py' is in the same directory as this script, or update BLENDER_SCRIPT_PATH.")
        exit()

    # Get prompt from user or use a default
    user_text_prompt = input("Enter your text prompt to generate a 3D model image: ")
    # if not user_text_prompt:
    # user_text_prompt = "a cube with a cylinder on top of it"
    output_file = "generated_image_for_prompt.png"

    print(f"\n--- Starting Text-to-Image Generation for input text ---")
    final_image = text_to_image_pipeline(
        user_text_prompt, model, tokenizer, final_output_image_path=output_file
    )
    if final_image:
        print(f"\n--- Pipeline Succeeded ---")
        print(f"Generated image saved at: {os.path.abspath(final_image)}")
    else:
        print(f"\n--- Pipeline Failed ---")
        print("Please check the logs above for errors.")
       