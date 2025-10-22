import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")  # Should return True
print(f"GPU device name: {torch.cuda.get_device_name(0)}")  # Prints your GPU model (e.g., "RTX 3060")