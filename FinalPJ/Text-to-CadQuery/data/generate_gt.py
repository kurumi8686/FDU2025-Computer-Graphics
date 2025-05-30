import os
import re
import cadquery as cq  # For the execution environment of the scripts
import traceback


def batch_process_cadquery_files_direct_replace():
    """
    Processes CadQuery Python files from "CQ",
    REPLACES occurrences of "./stlcq" (within quotes) with "../../stlcq",
    and executes them.
    This script should be run from the parent directory of "CQ" and "stlcq".
    """
    base_working_dir = os.getcwd()

    cq_source_root_dir = os.path.join(base_working_dir, "CQ")

    # This is where the script *expects* output based on your folder structure description
    intended_stlcq_output_root_dir = os.path.join(base_working_dir, "stlcq")

    if not os.path.isdir(cq_source_root_dir):
        print(f"Error: CadQuery source directory '{cq_source_root_dir}' not found.")
        print("Please ensure the 'CQ' directory exists and you are running this script from its parent directory.")
        return

    # We still create this directory structure as it was the user's stated goal for organization.
    # The actual output location of STLs will depend on how '../../stlcq' is resolved.
    os.makedirs(intended_stlcq_output_root_dir, exist_ok=True)

    print(f"Source CadQuery directory: {cq_source_root_dir}")
    print(f"--- IMPORTANT PATH WARNING ---")
    print(f"This script will replace string literals like './stlcq' with '../../stlcq'.")
    print(f"When the modified CadQuery scripts are executed, '../../stlcq' will be resolved relative to:")
    print(f"  {base_working_dir}")
    print(f"This means STLs will likely be written to a path like:")
    print(f"  {os.path.abspath(os.path.join(base_working_dir, '../../stlcq'))}")
    print(f"While this script will also ensure directories are created under:")
    print(f"  {intended_stlcq_output_root_dir}")
    print(f"Please VERIFY the actual STL output locations after running.")
    print(f"---------------------------------")

    processed_files_count = 0
    error_files_count = 0
    total_files_found = 0
    files_with_path_modification = 0

    for subdir_name in sorted(os.listdir(cq_source_root_dir)):
        current_cq_subdir = os.path.join(cq_source_root_dir, subdir_name)

        if os.path.isdir(current_cq_subdir):
            # Ensure the *intended* output sub-directory structure exists (for organizational reference)
            current_intended_stlcq_output_subdir = os.path.join(intended_stlcq_output_root_dir, subdir_name)
            os.makedirs(current_intended_stlcq_output_subdir, exist_ok=True)

            print(f"\nProcessing directory: {os.path.join('CQ', subdir_name)}")

            for filename in sorted(os.listdir(current_cq_subdir)):
                if filename.endswith(".py"):
                    total_files_found += 1
                    script_path = os.path.join(current_cq_subdir, filename)

                    print(f"  Processing file: {filename} ...")

                    try:
                        with open(script_path, 'r', encoding='utf-8') as f:
                            original_content = f.read()

                        modified_content = original_content

                        # 1. Remove visualization-related calls
                        modified_content = re.sub(r"^\s*from cadquery.vis import show.*?\n", "", modified_content,
                                                  flags=re.MULTILINE)
                        modified_content = re.sub(r"^\s*import cadquery.vis.*?\n", "", modified_content,
                                                  flags=re.MULTILINE)
                        modified_content = re.sub(r"^\s*show_object\s*\(.*?\)\s*\n?", "", modified_content,
                                                  flags=re.MULTILINE)
                        modified_content = re.sub(r"(\S+)\.show\s*\(.*?\)", r"\1", modified_content)
                        modified_content = re.sub(r"^\s*[^#\s][^#\n]*?\.show\s*\(.*?\)\s*\n?", "", modified_content,
                                                  flags=re.MULTILINE)

                        # 2. Perform the direct string replacement: './stlcq' -> '../../stlcq'
                        #    This regex specifically targets it within quotes.
                        #    G1: Opening quote (' or ")
                        #    G2: The "./stlcq" part
                        path_prefix_to_replace_pattern = re.compile(r"""(['"])(\./stlcq)""")

                        def replace_stlcq_prefix(match):
                            quote = match.group(1)
                            return f"{quote}../../stlcq"

                        temp_content = path_prefix_to_replace_pattern.sub(replace_stlcq_prefix, modified_content)

                        if temp_content != modified_content:
                            if processed_files_count == error_files_count:
                                files_with_path_modification += 1
                            print(f"    Replaced './stlcq' with '../../stlcq'")
                            modified_content = temp_content

                        exec_globals = {"cq": cq, "os": os}
                        exec(modified_content, exec_globals)

                        print(f"    Successfully processed and executed: {filename}")
                        processed_files_count += 1

                    except Exception as e:
                        print(f"    ERROR processing file {filename}: {type(e).__name__} - {e}")
                        error_files_count += 1

    print(f"\n--- Batch Processing Summary ---")
    print(f"Total Python files found: {total_files_found}")
    # The files_with_path_modification counter is tricky to get right without more state.
    # The printout during processing already indicates if a replacement was made for a file.
    print(f"Successfully processed files: {processed_files_count}")
    print(f"Files with errors during processing/execution: {error_files_count}")
    print("\nBatch processing finished.")
    print("Review the IMPORTANT PATH WARNING at the beginning of the script output regarding actual STL locations.")


if __name__ == '__main__':
    batch_process_cadquery_files_direct_replace()