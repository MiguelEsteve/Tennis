import os
import torch
from models.vision.flownet.flownet2S import FlowNet2S
from configs.configs import WEIGHTS_FLOWNET_PATH
from configs.log_conf import getLogger

LOGGER = getLogger(__name__)

class TestFlowNet2S:
    def __init__(self) -> None:
        self.model = FlowNet2S()

    def test_get_from_checkpoint(self):
        model = FlowNet2S.get_from_checkpoint()
        print(model)

    def test__call__(self):
        model = FlowNet2S.get_from_checkpoint()
        x_dummy = torch.rand(1,2, 3,384,512)
        pred = model.__call__(x_dummy)
        assert isinstance(pred, tuple)

    def test_predict(self):
        model = FlowNet2S.get_from_checkpoint()
        x_dummy = torch.rand(1,2, 3,384,512)
        pred = model.predict(x_dummy)
        print(f'pred shape: {pred.shape}')

    def test_forward(self):
        checkpoint_fn = os.path.join(WEIGHTS_FLOWNET_PATH, 'FlowNet2-S_checkpoint.pth')
        if not os.path.exists(checkpoint_fn):
            LOGGER.error(f'{checkpoint_fn} not found')
            return
        chkpt_dict = torch.load(checkpoint_fn)
        self.model.load_state_dict(chkpt_dict['state_dict'])
        
        x_dummy = torch.rand(1,3,384,512)
        pred = self.model(x_dummy)
        print(f'input shape to FlowNet2S: {x_dummy}')
        print(f'output shape of the model: {pred.shape}')
        


if __name__ == '__main__':
    t = TestFlowNet2S()
    # t.test_get_from_checkpoint()
    t.test__call__()
    # t.test_predict()
    # t.test_forward()