import torchvision
import torch

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
hh = 1