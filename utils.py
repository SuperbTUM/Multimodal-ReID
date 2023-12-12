import math
import pickle
import warnings
from functools import partial
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import model as clip_model
import custom_clip_model


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )


def resize_pos_embed(posemb, posemb_new, height, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[0]

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, height, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode='bicubic')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)
    return posemb


class BNNeck(nn.Module):
    def __init__(self, width, proj):
        super(BNNeck, self).__init__()
        if proj:
            self.bottleneck_proj = nn.BatchNorm1d(width)
            self.bottleneck_proj.bias.requires_grad_(False)
        else:
            self.bottleneck = nn.BatchNorm1d(width)
            self.bottleneck.bias.requires_grad_(False)
        self.proj = proj

    def forward(self, x):
        if self.proj:
            return self.bottleneck_proj(x)
        return self.bottleneck(x)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def model_adaptor(model, height, width, weights=None):
    # if (height, width) != (224, 224):
    if weights is not None:
        weights = torch.load(weights)
    if isinstance(model.visual, (clip_model.VisionTransformer, custom_clip_model.VisionTransformer)):
        bottleneck_proj = BNNeck(model.visual.proj.size(1), True)
        bottleneck = BNNeck(model.visual.proj.size(0), False)

        matched_weights = OrderedDict()
        bottleneck_weights = OrderedDict()

        if weights is not None:
            vision_width = weights["image_encoder.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in weights.keys() if k.startswith("image_encoder.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = weights["image_encoder.conv1.weight"].shape[-1]
            embed_dim = weights["text_encoder.text_projection"].shape[1]
            h_resolution = height // vision_patch_size
            w_resolution = width // vision_patch_size

            # pretrained_weight = weights["image_encoder.positional_embedding"].data
            # if pretrained_weight.size() != model.visual.positional_embedding.size():
            #     posemb = resize_pos_embed(pretrained_weight, model.visual.positional_embedding, h_resolution,
            #                               w_resolution)
            #     model.visual.positional_embedding = nn.Parameter(posemb)
            for key in weights:
                if key.startswith("image_encoder"):
                    matched_key = ".".join(key.split(".")[1:])
                    matched_weights[matched_key] = weights[key].to(model.state_dict()["visual."+matched_key].dtype)
                elif "bottleneck" in key:
                    matched_key = key
                    bottleneck_weights[matched_key] = weights[key]

            model.visual = custom_clip_model.VisionTransformer(h_resolution, w_resolution, vision_patch_size, 16,
                                                               vision_width, vision_layers, vision_width // 64,
                                                               embed_dim)

        if matched_weights:
            model.visual.load_state_dict(matched_weights, strict=True)
            convert_weights(model.visual)
        if bottleneck_weights:
            bottleneck.load_state_dict(bottleneck_weights, strict=False)
            bottleneck_proj.load_state_dict(bottleneck_weights, strict=False)
    else:
        h_resolution = height // 16
        w_resolution = width // 16
        # pretrained_weight = weights["image_encoder.attnpool.positional_embedding"].data
        # if model.visual.attnpool.positional_embedding.size() != pretrained_weight.size():
        #     posemb = resize_pos_embed(pretrained_weight, model.visual.attnpool.positional_embedding, h_resolution, w_resolution)
        #     model.visual.attnpool.positional_embedding = nn.Parameter(posemb)

        matched_weights = OrderedDict()
        bottleneck_weights = OrderedDict()
        if weights is not None:
            vision_width = weights["image_encoder.layer1.0.conv1.weight"].shape[0]
            vision_layers = tuple([len(
                set(k.split(".")[2] for k in weights if k.startswith(f"image_encoder.layer{b}"))) for b in [1, 2, 3, 4]])
            embed_dim = weights["text_encoder.text_projection"].shape[1]
            for key in weights:
                if key.startswith("image_encoder"):
                    matched_key = ".".join(key.split(".")[1:])
                    matched_weights[matched_key] = weights[key].to(model.state_dict()["visual." + matched_key].dtype)
                elif "bottleneck" in key:
                    matched_key = key
                    bottleneck_weights[matched_key] = weights[key]
            model.visual = custom_clip_model.ModifiedResNet(vision_layers, embed_dim, vision_width * 32 // 64,
                                                            h_resolution*w_resolution, vision_width)
        bottleneck = BNNeck(model.visual.attnpool.k_proj.in_features, False)
        bottleneck_proj = BNNeck(model.visual.attnpool.c_proj.out_features, True)
        if matched_weights:
            model.visual.load_state_dict(matched_weights, strict=True)
            convert_weights(model.visual)
        if bottleneck_weights:
            bottleneck.load_state_dict(bottleneck_weights, strict=False)
            bottleneck_proj.load_state_dict(bottleneck_weights, strict=False)

    return model.cuda(), bottleneck.cuda(), bottleneck_proj.cuda()
