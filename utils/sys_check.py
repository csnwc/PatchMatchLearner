# sys_check.py
import sys
import json
import torch
import torchvision
import pytorch_lightning
import pl_bolts

print(f"sys.path:\n{json.dumps(sys.path, indent=4)}")

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"PyTorch-Ligning version: {pytorch_lightning.__version__}")
print(f"pl_bolts version: {pl_bolts.__version__}")

print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device: {torch.cuda.current_device()}")
    print(f"torch.cuda.device: {torch.cuda.device(0)}")
    print(f"torch.cuda.get_device_name: {torch.cuda.get_device_name(0)}")
else:
    print(f"no GPU")