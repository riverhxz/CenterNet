import torch

from models.decode import ctdet_decode
from models.losses import FocalLoss, NormRegL1LossNoMask
from trains.base_trainer import BaseTrainer
from utils.post_process import ctdet_post_process


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        #     self.crit_reg = RegL1Loss()

        self.crit_wh = NormRegL1LossNoMask()
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hm_loss += self.crit(outputs['hm'], batch['hm'])
        wh_loss += self.crit_wh(
            outputs["wh"]
            , batch["wh_mask"]
            , batch["ind"]
            , batch["wh"])
        loss = (
                opt.hm_weight * hm_loss
                + opt.wh_weight * wh_loss
            #           + opt.off_weight * off_loss
        )
        loss_stats = {'loss': loss
            , 'hm_loss': hm_loss,
                      'wh_loss': wh_loss
                      #                   , 'off_loss': off_loss
                      }
        return loss, loss_stats


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['needle'], batch['stack'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats


class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
        self.model_with_loss = ModleWithLoss(model, self.loss)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
