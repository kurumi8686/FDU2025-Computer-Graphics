import os
import re
import json
import cadquery as cq
import traceback

# Input file for ground truth
input_jsonl_file = '../test_filtered.jsonl'
# Output directory for ground truth STL files
output_gt_stl_dir = '../output_gt'
os.makedirs(output_gt_stl_dir, exist_ok=True)

print(f"Reading ground truth CadQuery code from: {input_jsonl_file}")
print(f"Saving generated STL files to: {output_gt_stl_dir}")


def extract_and_clean_cadquery_code(text):
    """
    从模型输出或ground truth中提取并清理 CadQuery Python 代码。
    它会查找并去除 '<|endoftext|>' 标记，并尝试只保留有效的 Python 代码。
    同时，它会移除不必要的导入和硬编码的导出语句。
    """
    if "<|endoftext|>" in text: # May not be present in ground truth, but harmless
        text = text.split("<|endoftext|>")[0].strip()

    if "### Response:" in text: # May not be present in ground truth, but harmless
        text = text.split("### Response:", 1)[1].strip()

    # 简单清理：删除常见的 markdown 标记
    text = text.replace("```python", "").replace("```", "").strip()

    # 按行分割文本，以便逐行处理
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # 移除 cadquery.vis 的导入
        if "from cadquery.vis import show" in line:
            continue
        # 移除对 show() 函数的调用（如果模型或GT有生成的话）
        if "show(" in line:
            continue
        # 移除模型或GT中可能存在的硬编码的 cq.exporters.export() 调用
        if re.search(r'cq\.exporters\.export\(.*?\)', line):
            continue

        cleaned_lines.append(line)

    # 重新组合清理后的代码，并移除可能产生的空行
    return "\n".join(filter(None, cleaned_lines)).strip()


def run_cadquery_and_save_stl(code_str, output_path):
    """
    执行 CadQuery 代码并将其结果保存为 STL 文件。
    """
    try:
        # 使用 globals() 和 locals() 隔离执行环境
        exec_globals = {'cq': cq}
        exec_locals = {}
        exec(code_str, exec_globals, exec_locals)

        # 尝试从执行环境中找到 Workplane 对象
        # Common variable names for the final result in CadQuery scripts
        possible_result_vars = ['result', 'final_model', 'shape', 'model', 'part']
        result_obj = None

        # Check predefined common names first
        for var_name in possible_result_vars:
            if var_name in exec_locals and isinstance(exec_locals[var_name], cq.Workplane):
                result_obj = exec_locals[var_name]
                break
            elif var_name in exec_locals and isinstance(exec_locals[var_name], cq.Assembly):
                result_obj = exec_locals[var_name]
                break
            elif var_name in exec_locals and isinstance(exec_locals[var_name], cq.Shape): # Handle cq.Shape directly
                 # If it's a cq.Shape, we might need to put it on a workplane to use .val() or handle directly
                 # For simplicity, we assume cq.Shape can be exported if it's the intended final object.
                 # However, exportStl is typically a method of Solid or Compound.
                 # A cq.Workplane object's .val() usually returns a Solid or Compound.
                 # If the result is a raw cq.Shape that isn't a Solid/Compound, exportStl might fail.
                 # Let's try to wrap it if it's a Shape and not a Workplane/Assembly
                 pass # Covered by the generic loop below

        if not result_obj: # If not found by common names, scan all Workplane/Assembly/Shape objects
            for k, v in exec_locals.items():
                if isinstance(v, cq.Workplane) or isinstance(v, cq.Assembly):
                    result_obj = v
                    break
                # If the script directly creates a cq.Shape and assigns it, e.g. `my_shape = cq.Solid.makeBox(1,1,1)`
                # we need to handle this. cq.Workplane.val() returns a Shape (often Solid or Compound).
                # cq.Assembly can also be exported.
                if isinstance(v, cq.Shape) and not result_obj: # Prioritize Workplane/Assembly
                    result_obj = v


        if result_obj:
            if isinstance(result_obj, cq.Workplane):
                obj_to_export = result_obj.val() # Get the solid/compound from Workplane
            elif isinstance(result_obj, cq.Assembly):
                obj_to_export = result_obj.toCompound() # Assemblies can be exported
            elif isinstance(result_obj, cq.Shape): # e.g. cq.Solid, cq.Compound
                obj_to_export = result_obj
            else:
                return False, "Found object is not a Workplane, Assembly, or Shape."

            cq.exporters.export(obj_to_export, output_path)
            return True, f"Successfully generated STL: {output_path}"
        else:
            return False, "No CadQuery Workplane, Assembly, or Shape object found in the executed code."

    except Exception as e:
        return False, f"Error executing CadQuery code or saving STL: {e}\n{traceback.format_exc()}"


# Initialize a counter for output STL files
stl_counter = 0

# Check if the input file exists
if not os.path.exists(input_jsonl_file):
    print(f"Error: Input file {input_jsonl_file} not found.")
else:
    # Read the .jsonl file line by line
    with open(input_jsonl_file, 'r', encoding='utf-8') as f_in:
        for line_number, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line {line_number + 1} due to JSON decoding error: {e}")
                continue

            if "output" not in data:
                print(f"Skipping line {line_number + 1}: 'output' key not found in JSON object.")
                continue

            raw_cadquery_code = data["output"]
            output_stl_path = os.path.join(output_gt_stl_dir, f"{stl_counter}.stl")

            # Extract and clean CadQuery code
            cleaned_code = extract_and_clean_cadquery_code(raw_cadquery_code)

            if not cleaned_code:
                print(f"Skipping entry {stl_counter} (line {line_number + 1}): No code after cleaning.")
                # Optionally, still increment counter or handle as a failed processing
                # For now, we just skip and don't increment stl_counter for empty code
                continue

            # Run CadQuery code and save STL
            success, message = run_cadquery_and_save_stl(cleaned_code, output_stl_path)

            if success:
                print(f"Processed entry {stl_counter} (from line {line_number + 1}): {message}")
            else:
                print(f"Failed to process entry {stl_counter} (from line {line_number + 1}): {message}")

            stl_counter += 1 # Increment for the next file name

print("\nGround truth CadQuery code processing and STL generation complete.")