
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
cv2.warpAffine
from PIL import Image, ImageDraw,ImageFont
def get_pic_with_text(text,size=(128,128,3)):
    back_ground = np.ones(size,dtype=np.uint8) * 255
    img = Image.fromarray(back_ground)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10),text)
    ImageFont.truetype()
    draw.ellipse(((center[0]-10,center[1]-10),(center[0]+10,center[1]+10)),fill=0)
    img.paste()
    np.zeros()
    return img

import numpy as np

import  torch.nn.functional as F
F.l1_loss()
# import random
# random.randint()
#

#
# from models.losses import FocalLoss
# from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
# from models.decode import ctdet_decode
# from models.utils import _sigmoid
# from utils.debugger import Debugger
# from utils.post_process import ctdet_post_process
# from utils.oracle_utils import gen_oracle_map
# from .base_trainer import BaseTrainer
#
#
# class CtdetLoss(torch.nn.Module):
#     def __init__(self, opt):
#         super(CtdetLoss, self).__init__()
#         self.crit = FocalLoss()
#         self.crit_reg = RegLoss()
#         self.crit_wh = torch.nn.L1Loss(reduction='sum')
#         self.opt = opt
#
#     def forward(self, outputs, batch):
#         opt = self.opt
#         hm_loss, wh_loss, off_loss = 0, 0, 0
#         for s in range(opt.num_stacks):
#             output = outputs[s]
#             if not opt.mse_loss:
#                 output['hm'] = _sigmoid(output['hm'])
#
#             if opt.eval_oracle_hm:
#                 output['hm'] = batch['hm']
#             if opt.eval_oracle_wh:
#                 output['wh'] = torch.from_numpy(gen_oracle_map(
#                     batch['wh'].detach().cpu().numpy(),
#                     batch['ind'].detach().cpu().numpy(),
#                     output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
#             if opt.eval_oracle_offset:
#                 output['reg'] = torch.from_numpy(gen_oracle_map(
#                     batch['reg'].detach().cpu().numpy(),
#                     batch['ind'].detach().cpu().numpy(),
#                     output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)
#
#             hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
#             if opt.wh_weight > 0:
#                 if opt.dense_wh:
#                     mask_weight = batch['dense_wh_mask'].sum() + 1e-4
#                     wh_loss += (
#                                        self.crit_wh(output['wh'] * batch['dense_wh_mask'],
#                                                     batch['dense_wh'] * batch['dense_wh_mask']) /
#                                        mask_weight) / opt.num_stacks
#                 elif opt.cat_spec_wh:
#                     wh_loss += self.crit_wh(
#                         output['wh'], batch['cat_spec_mask'],
#                         batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
#                 else:
#                     wh_loss += self.crit_reg(
#                         output['wh'], batch['reg_mask'],
#                         batch['ind'], batch['wh']) / opt.num_stacks
#
#             if opt.reg_offset and opt.off_weight > 0:
#                 off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
#                                           batch['ind'], batch['reg']) / opt.num_stacks
#
#         loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
#                opt.off_weight * off_loss
#         loss_stats = {'loss': loss, 'hm_loss': hm_loss,
#                       'wh_loss': wh_loss, 'off_loss': off_loss}
#         return loss, loss_stats
import torch
import numpy as np
i = torch.ones(4,4).type(torch.LongTensor)
x=torch.zeros(4,4).gather(1,i)
print(x.shape)
from torch import nn
nn.AdaptiveMaxPool2d