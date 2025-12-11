import time
import numpy as np
import torch
import torch.jit # Needed for tracing, a key step in TensorRT preparation

# --- HELPER FUNCTIONS (Internal to this module) ---

def compile_to_tensorrt_fp16(model, dummy_input, device):
    """
    Converts the PyTorch model to half-precision (FP16) and traces it 
    to create a TorchScript module, simulating TensorRT graph building.
    """
    if device.type != 'cuda':
        # Compilation time is zero if CUDA is not available
        return model, 0.0

    # 1. Convert to Half Precision (FP16)
    model_fp16 = model.half()
    
    # Move the dummy input to FP16 as well
    dummy_input_fp16 = dummy_input.half().to(device)

    # 2. Trace the model (simulates building an optimized graph)
    start_compile_time = time.perf_counter()
    with torch.no_grad():
        # Trace the FP16 model with FP16 input
        traced_model = torch.jit.trace(model_fp16, dummy_input_fp16)
    
    compile_time_ms = (time.perf_counter() - start_compile_time) * 1000
    
    return traced_model, compile_time_ms


def benchmark_model(model, input_tensor, config, device):
    """
    Benchmarks the TensorRT-optimized model inference time.
    The model is assumed to be an FP16 JIT-traced CUDA model.
    """
    timings = []
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WARMUP = config['NUM_WARMUP']
    NUM_BENCHMARK = config['NUM_BENCHMARK']
    
    if device.type != 'cuda':
        # Non-optimized CPU path (should generally be skipped for TRT)
        mean_time_ms = 1000.0 # Placeholder for non-optimized CPU run
        median_latency = 1000.0
        p99_latency = 1000.0
        throughput_fps = 0.0
        
    else: # CUDA Timing Logic (for FP16/TensorRT Simulation)
        
        # The input tensor must be FP16 to match the compiled model
        input_tensor_fp16 = input_tensor.half().to(device) 
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(NUM_WARMUP):
                _ = model(input_tensor_fp16)
            torch.cuda.synchronize()
        
        # Measure performance
        with torch.no_grad():
            for _ in range(NUM_BENCHMARK):
                starter.record()
                _ = model(input_tensor_fp16)
                ender.record()
                torch.cuda.synchronize() # Wait for GPU
                
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time) # Time is in milliseconds (ms)

        # --- STATISTICAL CALCULATIONS ---
        timings_np = np.array(timings)
        
        mean_time_ms = timings_np.mean()
        median_latency = np.percentile(timings_np, 50)
        p99_latency = np.percentile(timings_np, 99)
        
        # METRIC 4: THROUGHPUT (FPS)
        throughput_fps = (BATCH_SIZE / mean_time_ms) * 1000


    return {
        'median_latency': median_latency,
        'p99_latency': p99_latency,
        'throughput_fps': throughput_fps,
    }


# -------------------------------------------------------------
# MAIN EXPORT FUNCTION (Called by master_comparison.py)
# -------------------------------------------------------------

def run_tensorrt_benchmark(config, torch_model, static_metrics):
    """
    Orchestrates the TensorRT (FP16) simulation benchmark.
    Returns a single standardized metric dictionary.
    """
    if not torch.cuda.is_available():
        print("TensorRT requires CUDA. Skipping TensorRT benchmark.")
        return []

    print("\n--- Starting TensorRT (FP16 Simulation) Benchmark ---")
    
    # 1. Setup Device and Input
    device = torch.device("cuda:0")
    # Dummy input must be FP32 initially before being passed to compilation
    dummy_input_fp32 = torch.randn(config['DUMMY_INPUT_SHAPE']).to(device)
    
    # 2. Compile to TensorRT/FP16 equivalent
    # Ensure the PyTorch model is on the device for compilation
    torch_model.to(device)
    tensorrt_model, compile_time_ms = compile_to_tensorrt_fp16(torch_model, dummy_input_fp32, device)

    # 3. Benchmarking
    dynamic_metrics = benchmark_model(tensorrt_model, dummy_input_fp32, config, device)
    
    # 4. Final Result Compilation
    
    # Get GPU name for the Target Hardware description
    device_name_str = f"{torch.cuda.get_device_name(0)} (TensorRT/FP16 Sim.)"
    
    # --- Build Standardized Dictionary ---
    # TensorRT optimization is primarily FP16 precision
    result = {
        'Framework': 'NVIDIA TensorRT',
        'Inf. Device': 'CUDA/GPU',
        'Target Hardware': device_name_str,
        'Deployed Precision': 'FP16',
        'Batch Size': config['BATCH_SIZE'],
        
        # Static Metrics (Pulled from static_setup.py)
        'Total Parameters': static_metrics['TOTAL_PARAMETERS'],
        # FP16 optimization halves the size of the FP32 model
        'Model Size (MB)': static_metrics['MODEL_SIZE_FP32_MB'] / 2, 
        'Observed PT Load Time (ms)': static_metrics['OBSERVED_PT_LOAD_TIME_MS'],
        
        # Dynamic Metrics
        'Compile/Export Time (ms)': compile_time_ms,
        'Avg. Latency (P50) (ms)': dynamic_metrics['median_latency'],
        'Worst-Case Latency (P99) (ms)': dynamic_metrics['p99_latency'],
        'Throughput (FPS)': dynamic_metrics['throughput_fps'],
    }
    
    print("--- TensorRT (FP16 Simulation) Benchmark Complete ---")
    return [result] # Return as a list for compatibility with the master script