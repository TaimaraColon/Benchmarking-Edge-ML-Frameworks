# The final script that calls everything
import pandas as pd
from IPython.display import display
import os
import torch
import numpy as np

# --- 1. IMPORT MODULAR COMPONENTS ---
from static_setup import get_static_metrics
from pt_benchmark import run_pytorch_benchmark
from trt_benchmark import run_tensorrt_benchmark
from ov_benchmark import run_openvino_benchmark
from tflite_benchmark import run_tflite_benchmark
from executorch_benchmark import run_executorch_benchmark


# -------------------------------------------------------------
# MASTER TABLE GENERATION FUNCTION
# -------------------------------------------------------------

def generate_final_comparison_table(results, config, static_metrics):
    """
    Consolidates all benchmark results into a single, comprehensive comparison table.
    """
    data = []
    
    # Format and consolidate data from the standardized result dictionaries
    for res in results:
        # Check if Compile/Export Time is a float (from a real benchmark) or a string (from Native)
        compile_time_val = res['Compile/Export Time (ms)']
        if isinstance(compile_time_val, float):
            compile_time_str = f"{compile_time_val:.2f}"
        else:
            compile_time_str = compile_time_val
            
        data.append({
            'Framework': res['Framework'],
            'Inf. Device': res['Inf. Device'],
            'Target Hardware': res['Target Hardware'],
            'Deployed Precision': res['Deployed Precision'],
            'Batch Size': res['Batch Size'],
            # Static Metrics (Formatted for display)
            'Model Size (MB)': f"{res['Model Size (MB)']:.2f}",
            'Total Parameters': f"{res['Total Parameters']:,}",
            'Observed PT Load Time (ms)': f"{res['Observed PT Load Time (ms)']:.2f}",
            # Dynamic Metrics (Formatted for display)
            'Compile/Export Time (ms)': compile_time_str,
            'Avg. Latency (P50) (ms)': f"{res['Avg. Latency (P50) (ms)']:.3f}",
            'Worst-Case Latency (P99) (ms)': f"{res['Worst-Case Latency (P99) (ms)']:.3f}",
            'Throughput (FPS)': f"{res['Throughput (FPS)']:.2f}",
        })
        
    df_final = pd.DataFrame(data)
    
    # Define the final column order for a clean presentation
    column_order = [
        'Framework', 'Inf. Device', 'Target Hardware', 'Deployed Precision',
        'Batch Size', 'Model Size (MB)', 'Total Parameters',
        'Observed PT Load Time (ms)', 'Compile/Export Time (ms)',
        'Avg. Latency (P50) (ms)', 'Worst-Case Latency (P99) (ms)',
        'Throughput (FPS)',
    ]
    
    df_final = df_final[column_order]
    
    # Ensure 'Throughput (FPS)' is numeric before sorting (needed because we formatted it as a string)
    # We must strip commas from the Total Parameters column to allow sorting if needed
    df_final['Throughput (FPS)'] = pd.to_numeric(df_final['Throughput (FPS)'])
    df_final_sorted = df_final.sort_values(by='Throughput (FPS)', ascending=False)

    print("\n\n#####################################################################")
    print("### FINAL FRAMEWORK COMPARISON ###")
    print("#####################################################################")
    print(f"\nModel: {static_metrics['MODEL_NAME']} | Total Parameters: {static_metrics['TOTAL_PARAMETERS']:,} | Batch Size: {config['BATCH_SIZE']}")
    print("\nTable sorted by Throughput (FPS) .")
    
    display(df_final_sorted)


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == '__main__':
    
    # --- 1. Static Setup (Run only once) ---
    print("--- 1. Running Static Setup (Model Loading and Config) ---")
    
    # The torch_model is an in-memory object passed to other functions.
    # The static_metrics dict contains load time, total parameters, and size.
    try:
        config, torch_model, static_metrics = get_static_metrics()
    except Exception as e:
        print(f"FATAL ERROR during static setup: {e}")
        print("Ensure 'static_setup.py' is correct and necessary libraries (torch, torchvision) are installed.")
        exit()

    print(f"Static Setup Complete. PT Model Load Time: {static_metrics['OBSERVED_PT_LOAD_TIME_MS']:.2f} ms")
    
    all_benchmark_results = []
    
    # --- 2. Run All Framework Functions Dynamically ---
    
    # NOTE: Each function returns a LIST of result dictionaries (e.g., CPU + GPU)
    
    print("\n\n--- 2. Starting Framework Benchmarks ---")
    
    # 2.1. PyTorch Native
    all_benchmark_results.extend(run_pytorch_benchmark(config, torch_model, static_metrics))
    
    # 2.2. TensorRT Simulation
    all_benchmark_results.extend(run_tensorrt_benchmark(config, torch_model, static_metrics))
    
    # 2.3. OpenVINO
    all_benchmark_results.extend(run_openvino_benchmark(config, torch_model, static_metrics))

    # 2.4. TFLite Simulation
    all_benchmark_results.extend(run_tflite_benchmark(config, torch_model, static_metrics))
    
    # 2.5. Executorch Simulation
    all_benchmark_results.extend(run_executorch_benchmark(config, torch_model, static_metrics))
    
    # master_comparison.py (Simplified code snippet, actual file contains the full logic)

    # --- 3. Generate Final Table ---
    print("\n\n--- 3. Aggregating Results and Generating Final Table ---")
    
    # Clean up large model objects to free memory
    del torch_model 
    
    if not all_benchmark_results:
        print("ERROR: No benchmark results were collected. Please check individual benchmark files for errors.")
    else:
        # FIX: Ensure 'config' is passed to the function, as it contains BATCH_SIZE and MODEL_NAME
        generate_final_comparison_table(all_benchmark_results, config, static_metrics)