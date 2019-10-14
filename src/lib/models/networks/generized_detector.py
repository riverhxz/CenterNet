from efficientnet_pytorch import EfficientNet
from torch import nn

from models.utils import _sigmoid

BN_MOMENTUM = 0.1
from torch import nn


class BackBone(nn.Module):
    def __init__(self, feature_map_dim, model, upconv=True):
        super(BackBone, self).__init__()
        self.deconv_with_bias = False
        self.inplanes = 1280
        self.model = model
        self.upconv = upconv
        if upconv:
            self.deconv_layers = self._make_deconv_layer(
                3,
                [256, 256, feature_map_dim],
                [4, 4, 4],
            )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.model.extract_features(x)
        if self.upconv:
            x = self.deconv_layers(x)

        return x


class GeneralizedDetector(nn.Module):
    def __init__(self, feature_map_dim=256):
        super(GeneralizedDetector, self).__init__()

        stack_model = EfficientNet.from_pretrained('efficientnet-b0')
        needle_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.stack_net = BackBone(feature_map_dim=feature_map_dim, model=stack_model)
        self.needle_net = BackBone(feature_map_dim=feature_map_dim, upconv=False, model=needle_model)
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(1280, feature_map_dim, kernel_size=1, padding=0, bias=False)
            , nn.ReLU(inplace=True)
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(feature_map_dim, 2, kernel_size=1, padding=0, bias=True)
        )

        self.hm_head = nn.Sequential(
            nn.Conv2d(feature_map_dim, 1, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, needle, stack):
        stack = self.stack_net(stack)
        needle = self.needle_net(needle)
        needle = self.pooling(needle)
        needle = self.fc(needle)

        feature_map = stack * needle

        wh_head = self.wh_head(feature_map)
        hm_head = self.hm_head(feature_map)
        hm_head = _sigmoid(hm_head)
        return {
            "wh": wh_head
            , "hm": hm_head
        }
