from typing import Any
import torch
import torch.nn as nn
import os
from torch.nn import init

import math
import numpy as np
from configs.configs import WEIGHTS_FLOWNET_PATH
from configs.log_conf import getLogger

from .flownetS import FlowNetS

LOGGER = getLogger(__name__)

class FlowNet2S(FlowNetS):
    @classmethod
    def get_from_checkpoint(cls):
        checkpoint = os.path.join(WEIGHTS_FLOWNET_PATH, 'FlowNet2-S_checkpoint.pth')
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'{checkpoint}')
        chkpt_dict = torch.load(checkpoint)
        model = cls()
        model.load_state_dict(chkpt_dict['state_dict'])
        return model

    def __init__(self, batchNorm=False, div_flow=20):
        super(FlowNet2S, self).__init__(input_channels = 6, batchNorm=batchNorm)
        self.div_flow = div_flow

    def __call__(self, x) -> Any:
        return super(FlowNet2S, self).__call__(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return super(FlowNet2S, self).__call__(x)

    def forward(self, inputs):

        b, t, c, w, h = inputs.shape
        new_shape = (b, t*c, w, h)
        x_reshaped = torch.reshape(inputs, new_shape)

        out_conv1 = self.conv1(x_reshaped)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return self.upsample1(flow2*self.div_flow)