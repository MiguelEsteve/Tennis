import torch
from models.vision.flownet.flownetS import FlowNetS

class TestFlowNetS():
    def __init__(self):
        self.model = FlowNetS()

    def test_forward(self):
        x_dummy = torch.randn(1,12,384,512)
        pred = self.model(x_dummy)

if __name__ == '__main__':
    
    t = TestFlowNetS()
    t.test_forward()
