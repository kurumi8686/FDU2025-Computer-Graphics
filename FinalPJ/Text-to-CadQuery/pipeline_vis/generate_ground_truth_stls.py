import os
import json
import cadquery as cq
import traceback
from tqdm import tqdm

def extract_and_clean_cadquery_code(text: str) -> str:
    """
    提取和清理 CadQuery Python 代码
    """
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0].strip()
    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()

    # 移除 markdown 代码块标记
    text = text.replace("```python", "").replace("```", "").strip()
    
    # 清理不需要的行
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # 跳过可视化相关的导入和调用
        if "from cadquery.vis import show" in line or "import cadquery.vis" in line:
            continue
        if "show_object(" in line or ".show(" in line:
            continue
        # 移除 export 语句，我们会自己添加
        if "cq.exporters.export" in line or "exporters.export" in line:
            continue
        cleaned_lines.append(line)
    
    return "\n".join(filter(None, cleaned_lines)).strip()

def run_cadquery_and_save_stl(code_str: str, output_stl_path: str) -> tuple[bool, str]:
    """
    执行 CadQuery 代码并保存为 STL 文件
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
        
        # 执行代码
        exec_globals = {'cq': cq}
        exec_locals = {}
        exec(code_str, exec_globals, exec_locals)
        
        # 查找结果对象
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
        elif found_workplane:
            result_obj = found_workplane

        if result_obj:
            if isinstance(result_obj, cq.Assembly):
                # 处理 Assembly 对象
                if result_obj.objects:
                    try:
                        shapes_to_compound = [obj.val() for obj in result_obj.objects.values() 
                                            if hasattr(obj, 'val') and isinstance(obj.val(), cq.Shape) and obj.val().isValid()]
                        if not shapes_to_compound:
                            return False, "Assembly contains no valid shapes to export."
                        compound = cq.Compound.makeCompound(shapes_to_compound)
                        if compound and compound.isValid():
                            cq.exporters.export(compound, output_stl_path)
                        else:
                            # 回退方案：导出第一个有效形状
                            exported_fallback = False
                            for obj_item in result_obj.objects.values():
                                if hasattr(obj_item, 'val') and isinstance(obj_item.val(), cq.Shape) and obj_item.val().isValid():
                                    cq.exporters.export(obj_item.val(), output_stl_path)
                                    exported_fallback = True
                                    break
                            if not exported_fallback:
                                return False, "Assembly object found but no valid shape to export."
                    except Exception as assembly_export_e:
                        return False, f"Error exporting CadQuery Assembly: {assembly_export_e}"
                else:
                    return False, "CadQuery Assembly object found, but it's empty."

            elif isinstance(result_obj, cq.Workplane):
                # 处理 Workplane 对象
                solid_to_export = result_obj.val()
                if not solid_to_export or not solid_to_export.isValid():
                    return False, "CadQuery Workplane did not result in a valid solid."
                if isinstance(solid_to_export, cq.Compound) and not solid_to_export.Solids():
                    return False, "CadQuery Workplane resulted in an empty compound."
                cq.exporters.export(solid_to_export, output_stl_path)

            # 验证文件是否成功创建
            if os.path.exists(output_stl_path) and os.path.getsize(output_stl_path) > 0:
                return True, f"Successfully generated STL: {output_stl_path}"
            else:
                return False, "STL file was not created or is empty."
        else:
            return False, "No CadQuery Workplane or Assembly object found in the executed code."

    except Exception as e:
        return False, f"Error executing CadQuery code or saving STL: {e}\n{traceback.format_exc()}"

def generate_ground_truth_stls(jsonl_file_path: str, output_dir: str, max_samples: int = None):
    """
    从 JSONL 文件中读取 CadQuery 代码并生成真值 STL 文件
    """
    print(f"Generating ground truth STL files from: {jsonl_file_path}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    error_files = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing JSONL entries")):
            # 限制处理数量（用于测试）
            if max_samples is not None and line_idx >= max_samples:
                print(f"Reached maximum samples limit: {max_samples}")
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                if "output" not in data:
                    print(f"Warning: Line {line_idx} missing 'output' field, skipping.")
                    continue
                    
                cadquery_code = data["output"]
                cleaned_code = extract_and_clean_cadquery_code(cadquery_code)
                
                if not cleaned_code:
                    print(f"Warning: Line {line_idx} has empty code after cleaning, skipping.")
                    error_count += 1
                    error_files.append(f"{line_idx}.stl (empty code)")
                    continue
                
                # 生成 STL 文件路径
                stl_filename = f"{line_idx}.stl"
                stl_path = os.path.join(output_dir, stl_filename)
                
                # 执行 CadQuery 代码
                success, message = run_cadquery_and_save_stl(cleaned_code, stl_path)
                
                if success:
                    success_count += 1
                    if line_idx < 5:  # 只为前几个显示详细信息
                        print(f"✓ Generated: {stl_filename}")
                else:
                    error_count += 1
                    error_files.append(f"{stl_filename} ({message})")
                    if line_idx < 10:  # 只为前几个显示错误信息
                        print(f"✗ Failed {stl_filename}: {message}")
                        
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_idx} JSON decode error: {e}")
                error_count += 1
                error_files.append(f"{line_idx}.stl (JSON decode error)")
            except Exception as e:
                print(f"Error: Line {line_idx} unexpected error: {e}")
                error_count += 1
                error_files.append(f"{line_idx}.stl (unexpected error)")
    
    print(f"\n=== Ground Truth STL Generation Summary ===")
    print(f"Successfully generated: {success_count} STL files")
    print(f"Failed: {error_count} files")
    print(f"Output directory: {output_dir}")
    
    if error_files and len(error_files) <= 20:  # 只显示前20个错误
        print(f"\nFirst few error files:")
        for error_file in error_files[:10]:
            print(f"  - {error_file}")
    
    return success_count, error_count

if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    JSONL_FILE = os.path.join(PROJECT_ROOT, 'inference/test_filtered.jsonl')
    GT_STL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'inference/output_gt_from_jsonl')
    
    # 生成真值 STL 文件
    # 设置 max_samples=10 用于测试，设置为 None 处理全部
    generate_ground_truth_stls(JSONL_FILE, GT_STL_OUTPUT_DIR, max_samples=10)