# Holds constants and the single PyTorch model loader
import time
import torch
import torchvision
from openvino.runtime import Core

# --- Global Configuration ---
BATCH_SIZE = 64
NUM_WARMUP = 10
NUM_BENCHMARK = 100
MODEL_NAME = "MobileNetV2"
DUMMY_INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224)

def get_static_metrics():
    """
    Loads the PyTorch model once, measures static metrics, and sets up configuration.
    """
    # 1. Load PyTorch Model & Measure Load Time
    start_load_time = time.perf_counter()
    torch_model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    )
    torch_model.eval()
    observed_pt_load_time_ms = (time.perf_counter() - start_load_time) * 1000

    # 2. Calculate Static Model Metrics
    total_params = sum(p.numel() for p in torch_model.parameters())
    model_size_fp32_mb = (total_params * 4) / (1024 * 1024)
    model_size_int8_mb = (total_params * 1) / (1024 * 1024)

    # 3. Determine Available OpenVINO Devices
    ov_core = Core()
    available_ov_devices = ["CPU"]
    if any("GPU" in dev for dev in ov_core.available_devices):
        available_ov_devices.append("GPU")

    # --- Configuration Dictionary ---
    config = {
        'BATCH_SIZE': BATCH_SIZE,
        'NUM_WARMUP': NUM_WARMUP,
        'NUM_BENCHMARK': NUM_BENCHMARK,
        'DUMMY_INPUT_SHAPE': DUMMY_INPUT_SHAPE,
        'AVAILABLE_OV_DEVICES': available_ov_devices,
    }

    # --- Static Metrics Dictionary ---
    static_metrics = {
        'MODEL_NAME': MODEL_NAME,
        'TOTAL_PARAMETERS': total_params,
        'MODEL_SIZE_FP32_MB': model_size_fp32_mb,
        'MODEL_SIZE_INT8_MB': model_size_int8_mb,
        'OBSERVED_PT_LOAD_TIME_MS': observed_pt_load_time_ms,
    }

    return config, torch_model, static_metrics