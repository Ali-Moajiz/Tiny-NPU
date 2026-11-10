"""Optional PyTorch template. Replace Net444 with EXACT training architecture.

Usage: python examples/export_pytorch.py trained_state_dict.pth model.onnx
PyTorch is an optional export dependency, not needed by the NPU compiler.
"""
import sys
import torch
from torch import nn


class Net444(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_pytorch.py state_dict.pth model.onnx")
    model = Net444()
    model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 4), sys.argv[2], opset_version=17,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      dynamo=False)
