import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont,ImageOps
import random
import os
from torch import nn
import cv2

from datasets.sample.char_gen import CTNumberDataset
from logger import Logger
from models.model import save_model, load_model
from models.networks.generized_detector import GeneralizedDetector
from trains.gdet import CtdetTrainer

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['CUDA_LAUNCH_BLOCKING']="1"


def main():
    def to_ncwh(x):
        return np.transpose(x, [2, 0, 1])

    def to_tensor(x):
        x = torch.from_numpy(x)
        return x

    def transform_by_keys(x, transform, keys):
        for k, v in x.items():
            if k in keys:
                x[k] = transform(v)
        return x

    import torchvision.transforms as transforms
    data_transform_composed = transforms.Compose([
        lambda x: transform_by_keys(x, to_ncwh, ["needle", "stack"])
        , lambda x: transform_by_keys(x, to_tensor, x.keys())
    ])

    def to_ncwh(x):
        return np.transpose(x, [2, 0, 1])

    def to_tensor(x):
        x = torch.from_numpy(x)
        return x

    def transform_by_keys(x, transform, keys):
        for k, v in x.items():
            if k in keys:
                x[k] = transform(v)
        return x

    import torchvision.transforms as transforms
    data_transform_composed = transforms.Compose([
        lambda x: transform_by_keys(x, to_ncwh, ["needle", "stack"])
        , lambda x: transform_by_keys(x, to_tensor, x.keys())
    ])

    from opts import opts
    opt = opts.init("")
    logger = Logger(opt)

    val_loader = torch.utils.data.DataLoader(
        CTNumberDataset(start=100000, length=10000, transform=data_transform_composed),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    train_loader = torch.utils.data.DataLoader(
        CTNumberDataset(0, 100000, transform=data_transform_composed),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    best = 1e10
    start_epoch = -1
    model = GeneralizedDetector()

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                logger.scalar_summary('val_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

