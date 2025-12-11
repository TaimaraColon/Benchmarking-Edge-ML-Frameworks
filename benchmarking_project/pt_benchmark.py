import time
import numpy as np
import torch

def run_pytorch_benchmark(config, torch_model, static_metrics):
    """
    Benchmarks the native PyTorch model on the available CPU and CUDA (if present).
    Returns a list of two standardized metric dictionaries.
    """
    
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_WARMUP = config['NUM_WARMUP']
    NUM_BENCHMARK = config['NUM_BENCHMARK']
    DUMMY_INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224)
    
    # 1. Prepare input tensor
    input_tensor = torch.randn(DUMMY_INPUT_SHAPE)
    
    # 2. Define target devices
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
        
    all_pt_results = []

    for device in devices_to_test:
        model = torch_model.to(device)
        input_data = input_tensor.to(device)
        
        timings = []
        
        # --- Run Benchmark Logic (Simplified from your original code) ---
        
        # Warmup
        for _ in range(NUM_WARMUP):
            with torch.no_grad():
                _ = model(input_data)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Measure
        for _ in range(NUM_BENCHMARK):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(input_data)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # ms
            
        timings_np = np.array(timings)
        mean_time_ms = timings_np.mean()
        median_latency = np.percentile(timings_np, 50)
        p99_latency = np.percentile(timings_np, 99)
        throughput_fps = (BATCH_SIZE / mean_time_ms) * 1000

        # --- Build Standardized Dictionary ---
        result = {
            'Framework': 'PyTorch (Native)',
            'Inf. Device': 'CUDA/GPU' if device == 'cuda' else 'CPU',
            'Target Hardware': 'Desktop/Server',
            'Deployed Precision': 'FP32',
            'Batch Size': BATCH_SIZE,
            'Total Parameters': static_metrics['TOTAL_PARAMETERS'],
            'Model Size (MB)': static_metrics['MODEL_SIZE_FP32_MB'],
            'Observed PT Load Time (ms)': static_metrics['OBSERVED_PT_LOAD_TIME_MS'],
            'Compile/Export Time (ms)': 'N/A (Native)',
            'Avg. Latency (P50) (ms)': median_latency,
            'Worst-Case Latency (P99) (ms)': p99_latency,
            'Throughput (FPS)': throughput_fps,
        }
        all_pt_results.append(result)
        
    return all_pt_results