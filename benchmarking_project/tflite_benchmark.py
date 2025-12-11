import time
import numpy as np
import os
# No pandas, display, or torchvision imports needed here—they are in the master script.
# The TFLite dependencies are still simulated, as in the original script.

# Define file path and device name globally for use in helper/main function
TFLITE_MODEL_PATH = "mobilenet_v2_fp32.tflite"
TFLITE_DEVICE_NAME = "Mobile/Embedded CPU"


# --- HELPER FUNCTIONS (Internal to this module) ---

def export_to_tflite(torch_model, static_metrics):
    """
    Simulates the export process from PyTorch (via ONNX) to TFLite.
    The torch_model is implicitly required for the actual export, though simulated here.
    """
    start_export_time = time.perf_counter()

    # Simulate a realistic export time based on model complexity
    # This time captures the graph parsing and conversion overhead (e.g., ONNX + TFLite conversion).
    # Use the static metrics passed from the master script
    simulated_export_time_ms = 450.0 + (static_metrics['TOTAL_PARAMETERS'] / 100000) * 1.5
    export_time_ms = (time.perf_counter() - start_export_time) * 1000 + simulated_export_time_ms

    print(f"Model successfully converted (simulated) to TFLite format: {TFLITE_MODEL_PATH}")

    return export_time_ms


def _run_tflite_benchmark(config):
    """
    Simulates the benchmark using the TFLite Interpreter.
    """
    timings = []
    
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WARMUP = config['NUM_WARMUP']
    NUM_BENCHMARK = config['NUM_BENCHMARK']
    
    # Create NumPy input data for the TFLite Interpreter (input is always a NumPy array)
    input_data = np.random.randn(*config['DUMMY_INPUT_SHAPE']).astype(np.float32)

    print(f"\n--- Starting TFLite Benchmark Simulation on {TFLITE_DEVICE_NAME} ---")

    # --- Warm-up (Simulated) ---
    simulated_warmup_latency = 15.0 # ms/batch
    for _ in range(NUM_WARMUP):
        time.sleep(simulated_warmup_latency / 1000) # Simulate time

    # --- Benchmark (Simulated) ---
    # Simulate a realistic optimized CPU inference time.
    base_simulated_latency = 12.0 # ms/batch on a strong mobile CPU for MobileNetV2

    for _ in range(NUM_BENCHMARK):
        start_time = time.perf_counter()
        time.sleep(base_simulated_latency / 1000) # Simulate the inference call
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)

    # --- STATISTICAL CALCULATIONS ---
    timings_np = np.array(timings)
    median_latency = np.percentile(timings_np, 50)
    p99_latency = np.percentile(timings_np, 99)
    # Use median for FPS calculation
    throughput_fps = (BATCH_SIZE / median_latency) * 1000

    return {
        'median_latency': median_latency,
        'p99_latency': p99_latency,
        'throughput_fps': throughput_fps,
    }


# -------------------------------------------------------------
# MAIN EXPORT FUNCTION (Called by master_comparison.py)
# -------------------------------------------------------------

def run_tflite_benchmark(config, torch_model, static_metrics):
    """
    Runs the TFLite export simulation and benchmark simulation.
    Returns a single standardized metric dictionary.
    """
    print("\n--- Starting TFLite Benchmarks---")
    
    # 1. Export Model (Performance Estimate)
    export_time_ms = export_to_tflite(torch_model, static_metrics)

    # 2. Run TFLite Benchmark Simulation
    dynamic_metrics = _run_tflite_benchmark(config)
    
    # 3. Clean up simulation files
    if os.path.exists(TFLITE_MODEL_PATH):
        os.remove(TFLITE_MODEL_PATH)
    if os.path.exists("mobilenet_v2.onnx"):
        os.remove("mobilenet_v2.onnx")

    # --- Build Standardized Dictionary ---
    # TFLite is typically optimized for INT8 on embedded systems
    result = {
        'Framework': 'TFLite (Performance Estimate)',
        'Inf. Device': 'Embedded CPU',
        'Target Hardware': TFLITE_DEVICE_NAME,
        'Deployed Precision': 'INT8',
        'Batch Size': config['BATCH_SIZE'],
        
        # Static Metrics (Pulled from static_setup.py)
        'Total Parameters': static_metrics['TOTAL_PARAMETERS'],
        'Model Size (MB)': static_metrics['MODEL_SIZE_INT8_MB'], # Use INT8 size
        'Observed PT Load Time (ms)': static_metrics['OBSERVED_PT_LOAD_TIME_MS'],
        
        # Dynamic Metrics
        'Compile/Export Time (ms)': export_time_ms,
        'Avg. Latency (P50) (ms)': dynamic_metrics['median_latency'],
        'Worst-Case Latency (P99) (ms)': dynamic_metrics['p99_latency'],
        'Throughput (FPS)': dynamic_metrics['throughput_fps'],
    }
    
    print("--- TFLite Benchmarks Complete ---")
    return [result] # Return as a list for compatibility with the master script