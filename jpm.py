import copy
import torch
import torch.nn as nn


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


class JPM(nn.Module):
    def __init__(self, last_block: nn.Module, last_layer_norm: nn.Module):
        super(JPM, self).__init__()
        self.last_block = nn.Sequential(
            copy.deepcopy(last_block),
            copy.deepcopy(last_layer_norm)
        )

    def forward(self, image_features_last):
        token = image_features_last[:, 0]
        image_features_no_token = image_features_last[:, 1:]
        image_features_shuffled = shuffle_unit(image_features_no_token, 5, 1)
        image_features_shuffled = self.last_block(torch.cat((token, image_features_shuffled), dim=1))
        return image_features_shuffled

