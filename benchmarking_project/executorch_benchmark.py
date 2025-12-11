import time
import numpy as np
import torch
import os

# Define file path globally for cleanup
EXECUTORCH_MODEL_PATH = "mobilenet_v2.pte"

# --- HELPER FUNCTIONS (Internal to this module) ---

def export_to_executorch(torch_model, config, static_metrics):
    """
    Simulates the export process from PyTorch to the Executorch format (.pte).
    This process is complex, involving graph lowering and serialization.
    """
    
    # Use the shape from the config
    DUMMY_INPUT = torch.randn(config['DUMMY_INPUT_SHAPE'])

    start_export_time = time.perf_counter()

    # Simulate a realistic export time based on model complexity (as in the original)
    simulated_export_time_ms = 600.0 + (static_metrics['TOTAL_PARAMETERS'] / 100000) * 2.5
    export_time_ms = (time.perf_counter() - start_export_time) * 1000 + simulated_export_time_ms

    print(f"Model successfully converted (simulated) to Executorch format: {EXECUTORCH_MODEL_PATH}")

    return export_time_ms


def _run_executorch_benchmark(config):
    """
    Simulates the benchmark using the Executorch Runtime.
    """
    timings = []
    
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WARMUP = config['NUM_WARMUP']
    NUM_BENCHMARK = config['NUM_BENCHMARK']
    
    # Create NumPy input data
    input_data = np.random.randn(*config['DUMMY_INPUT_SHAPE']).astype(np.float32)
    
    EXECUTORCH_DEVICE_NAME = "Mobile/Embedded CPU (XNNPACK)"
    
    print(f"\n--- Starting Executorch Benchmark Simulation on {EXECUTORCH_DEVICE_NAME} ---")

    # --- Warm-up (Simulated) ---
    simulated_warmup_latency = 13.0
    for _ in range(NUM_WARMUP):
        time.sleep(simulated_warmup_latency / 1000)
    
    # --- Benchmark (Simulated) ---
    base_simulated_latency = 10.0 # ms/batch on a strong mobile CPU for MobileNetV2

    for _ in range(NUM_BENCHMARK):
        start_time = time.perf_counter()
        # Simulate the inference call
        time.sleep(base_simulated_latency / 1000)
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)

    # --- STATISTICAL CALCULATIONS ---
    timings_np = np.array(timings)
    median_latency = np.percentile(timings_np, 50)
    p99_latency = np.percentile(timings_np, 99)
    throughput_fps = (BATCH_SIZE / median_latency) * 1000 # Use median for FPS calculation

    return {
        'median_latency': median_latency,
        'p99_latency': p99_latency,
        'throughput_fps': throughput_fps,
    }


# -------------------------------------------------------------
# MAIN EXPORT FUNCTION (Called by master_comparison.py)
# -------------------------------------------------------------

def run_executorch_benchmark(config, torch_model, static_metrics):
    """
    Runs the Executorch export simulation and benchmark simulation.
    Returns a single standardized metric dictionary.
    """
    print("\n--- Starting Executorch Benchmarks ---")
    
    # 1. Export Model (Simulated)
    export_time_ms = export_to_executorch(torch_model, config, static_metrics)

    # 2. Run Executorch Benchmark Simulation
    dynamic_metrics = _run_executorch_benchmark(config)

    # 3. Clean up simulation files
    if os.path.exists(EXECUTORCH_MODEL_PATH):
        os.remove(EXECUTORCH_MODEL_PATH)
        
    EXECUTORCH_DEVICE_NAME = "Mobile/Embedded CPU (XNNPACK)"

    # --- Build Standardized Dictionary ---
    # Executorch is typically optimized for INT8 on embedded systems
    result = {
        'Framework': 'Executorch (Performance Estimate)',
        'Inf. Device': 'Embedded CPU',
        'Target Hardware': EXECUTORCH_DEVICE_NAME,
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
    
    print("--- Executorch Benchmarks Complete ---")
    return [result] # Return as a list for compatibility with the master script