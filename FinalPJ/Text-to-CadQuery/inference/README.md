# Inference and Evaluation

This folder contains a 5-step pipeline for evaluating finetuned text-to-CadQuery models. The workflow goes from CadQuery code generation to 3D model rendering and quantitative evaluation.

## Pipeline Overview

1. step1_generate_CadQuery/  
   Use finetuned models to generate CadQuery code from natural language prompts.

2. step2_clean_run_CadQuery/  
   Extract valid Python code from model outputs and execute it to generate STL files using CadQuery.

3. step3_rendering/  
   Render STL files into 2D images using Blender.  
   Note: You must install and configure Blender version 3.2.2 for compatibility.  
   You can follow the instructions from:  
   https://github.com/allenai/objaverse-xl/tree/main/scripts/rendering

4. step4_gemini_eval/  
   Use Google Gemini 2.0 Flash to evaluate rendered images of the 3D models.  
   You must provide your own Gemini API key in the script or environment.

5. step5_compute_metrics/  
   Compute quantitative metrics such as Chamfer Distance and other 3D geometric similarity scores.

## Requirements

- Python environment with cadquery, trimesh, open3d, etc.
- Blender 3.2.2 installed and accessible via command line
- Valid Gemini API key for image-based evaluation

## Notes

- This pipeline supports batch processing of multiple models.
- Outputs include CadQuery code, rendered images, STL files, and evaluation metrics.
