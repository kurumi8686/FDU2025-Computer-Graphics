import os
import re
import cadquery as cq
import traceback

#input_dir = '../generated_cadquery'  # 第一步的输出目录
#output_stl_dir = '../generated_stl'  # STL 文件输出目录
input_dir = '../generated_cadquery_peft'  # 第一步的输出目录
output_stl_dir = '../generated_stl_peft'  # STL 文件输出目录

# 创建输出目录（如果不存在）
os.makedirs(output_stl_dir, exist_ok=True)

print(f"Reading generated CadQuery code from: {input_dir}")
print(f"Saving generated STL files to: {output_stl_dir}")


def extract_and_clean_cadquery_code(text):
    """
    从模型输出中提取并清理 CadQuery Python 代码。
    它会查找并去除 '<|endoftext|>' 标记，并尝试只保留有效的 Python 代码。
    同时，它会移除不必要的导入和硬编码的导出语句。
    """
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0].strip()

    if "### Response:" in text:
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
        # 移除对 show() 函数的调用（如果模型有生成的话）
        if "show(" in line:
            continue
        # 移除模型中可能存在的硬编码的 cq.exporters.export() 调用
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
        result = None
        for k, v in exec_locals.items():
            if isinstance(v, cq.Workplane):
                result = v
                break

        if result:
            result.val().exportStl(output_path)
            return True, f"Successfully generated STL: {output_path}"
        else:
            return False, "No CadQuery Workplane object found in the executed code."

    except Exception as e:
        return False, f"Error executing CadQuery code or saving STL: {e}\n{traceback.format_exc()}"


# 遍历 generated_cadquery 文件夹中的所有 txt 文件
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_dir, filename)
        base_name = os.path.splitext(filename)[0]
        output_stl_path = os.path.join(output_stl_dir, f"{base_name}.stl")

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # 提取并清理 CadQuery 代码
        cleaned_code = extract_and_clean_cadquery_code(raw_content)
        # 运行 CadQuery 代码并保存 STL
        success, message = run_cadquery_and_save_stl(cleaned_code, output_stl_path)
        if success:
            print(f"Processed {filename}: {message}")
        else:
            print(f"Failed to process {filename}: {message}")

print("\nCadQuery code cleaning and STL generation complete.")