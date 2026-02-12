import onnxruntime as ort
import sys
import torch
import os

def check_gpu():
    print(f"ðŸ” [SYSTEM] Checking Hardware Acceleration...")
    
    # 1. Check Torch (CUDA)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"âœ… [GPU] DETECTED: {gpu_name} ({vram:.2f} GB VRAM)")
    else:
        print("âŒ [GPU] CRITICAL ERROR: Torch cannot see your GPU.")
        # We don't exit here because ONNX might still work, but it's bad news.

    # 2. Check ONNX Runtime (The Real Worker)
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' not in providers:
        print("âŒ [ONNX] CRITICAL: CUDAExecutionProvider NOT found!")
        print(f"   Available: {providers}")
        print("   The system will run on CPU (SLOW).")
        return False
    else:
        print("âœ… [ONNX] CUDA Provider Available. AI will run on GPU.")
        return True
