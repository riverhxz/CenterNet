import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import os
from torch import nn
import cv2


def get_pic_with_text(text

                      , background
                      , mask
                      , fontsize=100
                      , offset=(0, 0)
                      , rotate=0
                      , font=None
                      ):
    # set parameter
    w = int(fontsize * 0.6)
    h = int(fontsize)
    x, y = offset
    center = [int(x + w / 2), int(y + h / 2)]
    patch_center = int(w / 2), int(h / 2)

    bw, bh = background.shape[:2]
    mw, mh = mask.shape[:2]
    scale_w, scale_h = 1.0 * bw / mw, 1.0 * bh / mh

    # initial initmage
    img = Image.fromarray(background)
    draw = ImageDraw.Draw(img)

    # draw patch
    patch = Image.new('L', (w, h))
    d = ImageDraw.Draw(patch)
    d.text((0, 0), text, font=font, fill=255)

    patch = patch.rotate(rotate, expand=1, center=patch_center)
    w, h = patch.size
    center = [int((x + w / 2) // scale_w), int((y + h / 2) // scale_h)]
    img.paste(ImageOps.colorize(patch, (0, 0, 0), (255, 255, 255)), offset, patch)

    ry, rx = int(patch.size[0] // 2 // scale_h), int(patch.size[1] // 2 // scale_w)
    radius = (ry, rx)

    mask_patch = get_mask((ry, rx))
    _mask = np.maximum(mask[center[1] - radius[1]:center[1] + radius[1], center[0] - radius[0]:center[0] + radius[0]],
                       mask_patch)
    mask[center[1] - rx:center[1] + rx, center[0] - ry:center[0] + ry] = _mask

    return np.asarray(img, dtype=np.uint8), mask, center, (w, h)


def random_position(image_size, fontsize):
    mx, my = image_size
    w = int(fontsize * 0.6)
    h = int(fontsize)
    min_x, min_y = 0, 0
    max_x, max_y = mx - w * 1.8, my - h * 1.8
    return random.randint(min_x, max_x), random.randint(min_y, max_y)


def random_char(char_set):
    return random.choice(char_set)


def get_mask(radius, scale=1):
    rx, ry = radius
    X = np.linspace(-rx, rx, 2 * rx)
    Y = np.linspace(-ry, ry, 2 * ry)
    X, Y = np.meshgrid(X, Y)

    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    Sigma = np.array([[scale * rx, 0], [0, scale * ry]])
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    def multivariate_gaussian(pos, mu, Sigma):
        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2 * np.pi) ** n * Sigma_det)

        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

        return np.exp(-fac / 2) / N

    # The distribution on the variables X, Y packed into pos.
    Z = multivariate_gaussian(pos, mu, Sigma)
    Z = Z / np.max(Z)
    return Z


def generate_pic(
        charSet
        , rotate
        , size
        , fontsize
        , font
        , num_char_per_pic=10
        , inv_mask_scale=4

):
    w, h = size[:2]
    img = np.ones(size, dtype=np.uint8) * 0
    mask = np.zeros((w // inv_mask_scale, h // inv_mask_scale, len(charSet)))
    center_list = []
    char_list = []
    wh_list = []
    for i, c in enumerate(charSet):
        whs = []
        center_clz = []
        for _ in range(num_char_per_pic):
            img, single_mask, center, wh = get_pic_with_text(
                text=c
                , background=img
                , mask=mask[..., i]
                , fontsize=fontsize
                , offset=random_position(size[:2], fontsize)
                , font=font
                , rotate=random.randint(rotate[0], rotate[1]))
            whs.append(wh)
            mask[..., i] = single_mask
            char_list.append(c)
            center_clz.append(center)

        center_list.append(center_clz)
        wh_list.append(whs)

    return img, mask, center_list, char_list, wh_list


'''
batch = {
,"needle":torch.ones(1,3,256,256).cuda()
"hm": torch.ones(1,1,64,64).cuda()
,'wh_mask': torch.ones(1,max_objects).cuda()
,'ind': torch.ones(1,max_objects).type('torch.LongTensor').cuda()
,'wh': torch.ones(1,max_objects,wh_dim).cuda()
}
'''


def generate_batch(
        #     batch_size=1
        size=[256, 256, 3]
        , needle_size=[100, 100, 3]
        , fontsize=50
        , charSet=["A", "B", "C"]
        , rotate=(-20, 20)
        , font=None
        , num_char_per_pic=3
        , inv_mask_scale=4
        , seed=None
):
    if seed:
        random.seed(seed)
    num_clz = len(charSet)
    char2idx = dict(zip(charSet, range(len(charSet))))

    stack, hm, center_list, char_list, whs = generate_pic(charSet, rotate=rotate, fontsize=fontsize, font=font,
                                                          size=size, num_char_per_pic=num_char_per_pic,
                                                          inv_mask_scale=inv_mask_scale)
    needle_char = random.choice(char_list)
    needle, _, _, _, _ = generate_pic(needle_char, rotate, size=needle_size, num_char_per_pic=1, font=font,
                                      fontsize=fontsize)
    needle_idx = char2idx[needle_char]

    #     wh_gt = torch.zeros([num_char_per_pic, 2,size[0],size[2]])
    #     for i,(wc,wd) in enumerate(batch2["pos"]):
    #         wh_gt[:,wc,wd] = batch2["wh"][i]

    pos = np.array(center_list[needle_idx])
    hm = np.array(hm[..., needle_idx], dtype=np.float32)
    return {
        "stack": np.array(stack, dtype=np.float32)  # 在图像中查找物体 w,h,c
        , "needle": np.array(needle, dtype=np.float32)  # 需要查找到物体图像 w,h,c
        , "hm": hm  # 中心点热力图
        , "wh_mask": np.array(np.arange(num_char_per_pic))  # 类别
        , 'pos': pos  # 中心点原始坐标
        , "ind": pos[..., 0] * hm.shape[1] + pos[..., 1]  # 中心点编码坐标
        , "wh": np.array(whs[needle_idx], dtype=np.float32)  # 宽度
        #         , "wh_gt": wh_gt
    }

def get_font(file, size=50):
    ImageFont.truetype(file, size)

import torch.utils.data as data


class CTNumberDataset(data.Dataset):
    num_classes = 3
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, start=0, length=100000, transform=None,
                  size=[256, 256, 3]
                 , needle_size=[100, 100, 3]
                 , fontsize=50
                 , charSet=["A", "B", "C"]
                 , rotate=(-20, 20)
                 , font=None
                 , num_char_per_pic=3
                 , inv_mask_scale=4
                 ):
        self.start = 0
        self.length = length
        self.transform = transform
        self.inputs = ["stack", "needle"]
        self.size = size
        self.needle_size = needle_size
        self.fontsize = fontsize
        self.charSet = charSet
        self.rotate = rotate
        self.font = font
        self.num_char_per_pic = num_char_per_pic
        self.inv_mask_scale = inv_mask_scale

    def __getitem__(self, index):
        x = generate_batch(
            size=self.size
            , needle_size=self.needle_size
            , fontsize=self.fontsize
            , charSet=self.charSet
            , rotate=self.rotate
            , font=self.font
            , num_char_per_pic=self.num_char_per_pic
            , inv_mask_scale=self.inv_mask_scale
            , seed=index + self.start)
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return self.length
