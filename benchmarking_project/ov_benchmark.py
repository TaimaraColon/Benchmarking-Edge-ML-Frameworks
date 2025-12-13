import time
import numpy as np
import pandas as pd
import torch
from openvino.runtime import Core, Tensor, serialize
from openvino import convert_model
import os
import torch.jit

# Initialize OpenVINO Core globally for helper functions
OV_CORE = Core()

# --- HELPER FUNCTIONS (Internal to this module) ---

def export_openvino_model(torch_model, config):
    """
    Converts the PyTorch model to OpenVINO IR and saves it.
    """
    MODEL_FP32_IR_PATH = "mobilenet_v2_fp32.xml"
    
    # 1. Prepare Model and Input on CPU and ensure FP32 (CRITICAL FIX)
    torch_model.cpu()
    # FIX: Explicitly cast the model's weights to FP32 (FloatTensor)
    torch_model.float() 
    
    # Input is already FloatTensor (FP32) by default from torch.randn
    dummy_input = torch.randn(config['DUMMY_INPUT_SHAPE']).cpu() 
    
    start_export_time = time.perf_counter()

    # --- FORCE PYTORCH TRACING ---
    with torch.no_grad():
        traced_model = torch.jit.trace(torch_model, dummy_input)
    
    # *** MODERN OPENVINO CONVERSION ***
    ov_model = convert_model(
        traced_model, 
        example_input=dummy_input,
    )

    # Serialize (Save) the model to OpenVINO IR
    serialize(
        ov_model,
        MODEL_FP32_IR_PATH,
        MODEL_FP32_IR_PATH.replace('.xml', '.bin')
    )

    export_time_ms = (time.perf_counter() - start_export_time) * 1000

    print("OpenVINO model successfully converted to IR format.")
    
    return ov_model, export_time_ms

def compile_openvino_model_for_device(ov_model, device_name):
    """
    Compiles the in-memory OpenVINO model for a specific device.
    """
    start_compile_time = time.perf_counter()

    # Read the model and compile for the specific device
    compiled_model = OV_CORE.compile_model(
        model=ov_model,
        device_name=device_name
    )

    compile_time_ms = (time.perf_counter() - start_compile_time) * 1000

    return compiled_model, compile_time_ms

def benchmark_openvino_model(compiled_model, device_name, config):
    """
    Benchmarks the compiled OpenVINO model.
    """
    timings = []
    
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WARMUP = config['NUM_WARMUP']
    NUM_BENCHMARK = config['NUM_BENCHMARK']

    # Create OpenVINO Input Tensor from NumPy
    input_data = np.random.randn(*config['DUMMY_INPUT_SHAPE']).astype(np.float32)
    input_tensor = Tensor(input_data)

    # Get the input layer name
    input_layer = compiled_model.input(0)
    input_name = input_layer.any_name

    # --- Warm-up ---
    for _ in range(NUM_WARMUP):
        compiled_model({input_name: input_tensor})

    # --- Measure performance ---
    for _ in range(NUM_BENCHMARK):
        start_time = time.perf_counter()
        compiled_model({input_name: input_tensor})
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000) # ms

    # --- STATISTICAL CALCULATIONS ---
    timings_np = np.array(timings)
    mean_time_ms = timings_np.mean()
    median_latency = np.percentile(timings_np, 50)
    p99_latency = np.percentile(timings_np, 99)

    # METRIC: THROUGHPUT (FPS)
    throughput_fps = (BATCH_SIZE / mean_time_ms) * 1000

    return {
        'median_latency': median_latency,
        'p99_latency': p99_latency,
        'throughput_fps': throughput_fps,
    }


# -------------------------------------------------------------
# MAIN EXPORT FUNCTION (Called by master_comparison.py)
# -------------------------------------------------------------

def run_openvino_benchmark(config, torch_model, static_metrics):
    """
    Orchestrates the OpenVINO benchmark across available devices.
    Returns a list of standardized metric dictionaries.
    """
    print("\n--- Starting OpenVINO Benchmarks ---")
    
    MODEL_FP32_IR_PATH = "mobilenet_v2_fp32.xml"
    all_ov_results = []

    # 1. Static Export
    try:
        ov_model, export_time_ms = export_openvino_model(torch_model, config)
    except Exception as e:
        print(f"Error during OpenVINO export: {e}. Skipping OpenVINO benchmarks.")
        return []
    
    # 2. Dynamic Compile and Benchmark (Loop through devices)
    for device in config['AVAILABLE_OV_DEVICES']:
        print(f"Benchmarking OpenVINO on device: {device}...")
        
        try:
            # 2a. Compile Model
            ov_compiled_model, compile_time_ms = compile_openvino_model_for_device(ov_model, device)
            
            # 2b. Benchmark
            dynamic_metrics = benchmark_openvino_model(ov_compiled_model, device, config)
            
            # --- Build Standardized Dictionary ---
            result = {
                'Framework': 'OpenVINO',
                'Inf. Device': device.upper(),
                'Target Hardware': 'Intel CPU/iGPU',
                'Deployed Precision': 'FP32',
                'Batch Size': config['BATCH_SIZE'],
                
                'Total Parameters': static_metrics['TOTAL_PARAMETERS'],
                'Model Size (MB)': static_metrics['MODEL_SIZE_FP32_MB'],
                'Observed PT Load Time (ms)': static_metrics['OBSERVED_PT_LOAD_TIME_MS'],
                
                'Compile/Export Time (ms)': compile_time_ms + export_time_ms,
                'Avg. Latency (P50) (ms)': dynamic_metrics['median_latency'],
                'Worst-Case Latency (P99) (ms)': dynamic_metrics['p99_latency'],
                'Throughput (FPS)': dynamic_metrics['throughput_fps'],
            }
            all_ov_results.append(result)
            
            del ov_compiled_model
            
        except Exception as e:
            print(f"Error benchmarking OpenVINO on {device}: {e}. Skipping device.")
            continue


    # 3. Clean up exported files
    if os.path.exists(MODEL_FP32_IR_PATH):
        os.remove(MODEL_FP32_IR_PATH)
    if os.path.exists(MODEL_FP32_IR_PATH.replace('.xml', '.bin')):
        os.remove(MODEL_FP32_IR_PATH.replace('.xml', '.bin'))
        
    print("--- OpenVINO Benchmarks Complete ---")
    return all_ov_results