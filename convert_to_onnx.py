import torch
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "brain_tumor_classifier.onnx")
