import torch
import torch_xla
import torch_xla.core.xla_model as xm

# 1. Fetch the default XLA device
device = xm.xla_device()
print(f"✅ Target XLA Device: {device}")

# 2. Check the hardware type behind the XLA device
device_type = xm.xla_device_hw(device)
print(f"🔥 Hardware Type: {device_type}") 

if device_type == "TPU":
    print("🎉 Success! PyTorch/XLA is successfully utilizing your TPU.")
else:
    print("❌ Warning: Hardware mapped to CPU/GPU. Check your TPU VM topology settings.")
