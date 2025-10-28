import time
import numpy as np
import torch
import torchvision
from PIL import Image
# Make sure to run: pip install git+https://github.com/modestyachts/ImageNetV2_pytorch.git in the terminal
from imagenetv2_pytorch import ImageNetV2Dataset 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Global Configuration ---
BATCH_SIZE = 64 
NUM_WARMUP = 10
NUM_BENCHMARK = 100
# Define the target device. We will assume CUDA is preferred if available.
TARGET_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# METRIC 1 & 2: MODEL LOAD TIME & MODEL SIZE CALCULATION
# -------------------------------------------------------------

# Start timer for model loading
start_load_time = time.perf_counter()

# Load MobileNetV2 model
# This is a large part of the load time calculation
torch_model = torchvision.models.mobilenet_v2(
    weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
)
torch_model.eval()

# Move model to the TARGET_DEVICE (e.g., CUDA)
torch_model.to(TARGET_DEVICE)

# Calculate load time
model_load_time_ms = (time.perf_counter() - start_load_time) * 1000

# Calculate model size (Static Metric)
total_params = sum(p.numel() for p in torch_model.parameters())
# Assuming FP32: 4 bytes per parameter. Convert to Megabytes.
model_size_mb = (total_params * 4) / (1024 * 1024)

# -------------------------------------------------------------

# Dataset and DataLoader setup
# Note: The DataLoader is set up but not used in the benchmark function itself.
dataset = ImageNetV2Dataset("matched-frequency")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Dummy input for benchmarking (Matches MobileNetV2 standard input size)
# Input is moved to the TARGET_DEVICE (e.g., CUDA)
dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(TARGET_DEVICE)

print("Model and data loader set up successfully.")

def benchmark_model(model, input_tensor, num_warmup, num_benchmark):
    """
    Benchmarks the model inference time, calculating full latency statistics.
    
    This function handles both CUDA and CPU timing accurately.
    """
    timings = [] # List to store all individual run times
    device_type = input_tensor.device.type
    
    # --- CUDA TIMING LOGIC ---
    if device_type == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        print(f"Warming up for {num_warmup} iterations on CUDA...")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(input_tensor)
        torch.cuda.synchronize()
        print("Warm-up complete. Starting benchmark...")
        
        with torch.no_grad():
            for rep in range(num_benchmark):
                starter.record()
                _ = model(input_tensor)
                ender.record()
                torch.cuda.synchronize() # Wait for GPU
                
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time) # Time is in milliseconds (ms)

    # --- CPU TIMING LOGIC ---
    else: 
        print(f"Warming up for {num_warmup} iterations on CPU...")
        
        # Ensure the model is on CPU if the device is CPU
        # We explicitly move model and input to CPU for a dedicated CPU test
        model_on_cpu = model.to('cpu') 
        input_on_cpu = input_tensor.to('cpu')
        
        # Warm-up
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = model_on_cpu(input_on_cpu) 
            
        print("Warm-up complete. Starting benchmark...")

        # Measure performance
        for rep in range(num_benchmark):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model_on_cpu(input_on_cpu)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000) # Time in milliseconds (ms)
            
        # Move the model back to the TARGET_DEVICE (optional, but good practice)
        model.to(TARGET_DEVICE)

    # --- ALL STATISTICAL CALCULATIONS (Common to both CUDA and CPU) ---
    mean_time_ms = sum(timings) / len(timings)
    std_time_ms = torch.tensor(timings).std().item()
    
    # METRIC 3: MEDIAN/PERCENTILE LATENCY
    median_latency = np.percentile(timings, 50)
    p90_latency = np.percentile(timings, 90)
    p99_latency = np.percentile(timings, 99)
    
    # METRIC 4: THROUGHPUT (FPS)
    throughput_fps = 1000.0 / mean_time_ms

    print(f"\n--- Benchmark Results ({device_type.upper()} @ BATCH={input_tensor.shape[0]}) ---")
    print(f"Inference Time (Avg over {num_benchmark} runs): {mean_time_ms:.3f} ms")
    print(f"Standard Deviation: {std_time_ms:.3f} ms")
    print(f"Median Latency (P50): {median_latency:.3f} ms")
    print(f"P90 Latency: {p90_latency:.3f} ms")
    print(f"P99 Latency: {p99_latency:.3f} ms")
    print(f"Throughput (FPS): {throughput_fps:.2f} FPS")


if __name__ == '__main__':
    # Print the static metrics first
    print(f"\n--- Model Static Metrics (MobileNetV2) ---")
    print(f"Model Load Time: {model_load_time_ms:.3f} ms")
    print(f"Model Size (FP32): {model_size_mb:.2f} MB")
    print(f"Total Parameters: {total_params:,}")
    print("-----------------------------------------")
    
    # Using dummy input is the required best practice to ensure a fair and rigorous comparison.
    # The reason a dummy input is better for this specific goal is that it completely isolates 
    # the model's pure computation time from all other bottlenecks.
    benchmark_model(torch_model, dummy_input, NUM_WARMUP, NUM_BENCHMARK)