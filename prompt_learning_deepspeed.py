import os
import json
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import deepspeed
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from tqdm import tqdm

from utils import load_pretrained_weights
from data_prepare import get_loader_train
from losses import SupConLoss, WeightedRegularizedTriplet
from schedulers import ConstantWarmupScheduler, create_scheduler

cudnn.enabled = True
cudnn.deterministic = True

from coop import build_model as build_model_coop, PromptLearner as PromptLearnerCoop
from cocoop import build_model as build_model_cocoop, PromptLearner as PromptLearnerCoCoop, TextEncoder
from maple import build_model as build_model_maple, MultiModalPromptLearner, TextEncoder as TextEncoderMaple
import clip_maple


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class CustomCLIPCoop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearnerCoop(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.vision_bottleneck = nn.BatchNorm1d(768)
        self.vision_bottleneck.bias.requires_grad_(False)
        self.vision_bottleneck.apply(weights_init_kaiming)
        self.vision_classifier = nn.Linear(768, classnames, bias=False)
        self.vision_classifier.apply(weights_init_classifier)

        self.vision_bottleneck_proj = nn.BatchNorm1d(512)
        self.vision_bottleneck_proj.bias.requires_grad_(False)
        self.vision_bottleneck_proj.apply(weights_init_kaiming)
        self.vision_classifier_proj = nn.Linear(512, classnames, bias=False)
        self.vision_classifier_proj.apply(weights_init_classifier)

    def forward(self, image=None, label=None, get_image=False, get_texts=False):
        if get_texts:
            prompts = self.prompt_learner(label)
            tokenized_prompts = self.tokenized_prompts

            text_features = self.text_encoder(prompts, tokenized_prompts)
            return text_features

        if get_image:
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features[:, 0]
            return image_features

        image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
        image_features_last = image_features_last[:, 0]
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]

        image_features_non_proj = self.vision_bottleneck(image_features_non_proj)
        cls_score = self.vision_classifier(image_features_non_proj.float())
        image_features = self.vision_bottleneck_proj(image_features)
        cls_score_proj = self.vision_classifier_proj(image_features.float())

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(label)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.t()

        if self.training:
            return text_features, image_features, logits, [cls_score, cls_score_proj], [image_features_last,
                                                                                        image_features_non_proj,
                                                                                        image_features]
        else:
            return torch.cat((image_features_non_proj, image_features), dim=1)


class CustomCLIPCoCoop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearnerCoCoop(classnames, clip_model, params)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.vision_bottleneck = nn.BatchNorm1d(768)
        self.vision_bottleneck.bias.requires_grad_(False)
        self.vision_bottleneck.apply(weights_init_kaiming)
        self.vision_classifier = nn.Linear(768, classnames, bias=False)
        self.vision_classifier.apply(weights_init_classifier)

        self.vision_bottleneck_proj = nn.BatchNorm1d(512)
        self.vision_bottleneck_proj.bias.requires_grad_(False)
        self.vision_bottleneck_proj.apply(weights_init_kaiming)
        self.vision_classifier_proj = nn.Linear(512, classnames, bias=False)
        self.vision_classifier_proj.apply(weights_init_classifier)

    def forward(self, image, label=None, get_image=False, get_texts=False):
        if get_image:
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features[:, 0]
            return image_features

        image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
        image_features_last = image_features_last[:, 0]
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]

        image_features_non_proj = self.vision_bottleneck(image_features_non_proj)
        cls_score = self.vision_classifier(image_features_non_proj.float())
        image_features = self.vision_bottleneck_proj(image_features)
        cls_score_proj = self.vision_classifier_proj(image_features.float())

        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(image_features, label)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        logits = []
        texts = []

        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
            texts.append(text_features)

        logits = torch.stack(logits)
        texts = torch.stack(texts)

        if get_texts:
            return texts

        if self.training:
            return texts, image_features, logits, [cls_score, cls_score_proj], [image_features_last,
                                                                                image_features_non_proj, image_features]
        else:
            return torch.cat((image_features_non_proj, image_features), dim=1)


class CustomCLIPMaple(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoderMaple(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.vision_bottleneck = nn.BatchNorm1d(768)
        self.vision_bottleneck.bias.requires_grad_(False)
        self.vision_bottleneck.apply(weights_init_kaiming)
        self.vision_classifier = nn.Linear(768, classnames, bias=False)
        self.vision_classifier.apply(weights_init_classifier)

        self.vision_bottleneck_proj = nn.BatchNorm1d(512)
        self.vision_bottleneck_proj.bias.requires_grad_(False)
        self.vision_bottleneck_proj.apply(weights_init_kaiming)
        self.vision_classifier_proj = nn.Linear(512, classnames, bias=False)
        self.vision_classifier_proj.apply(weights_init_classifier)

    def forward(self, image=None, label=None, get_image=False, get_texts=False):
        tokenized_prompts = self.tokenized_prompts

        if get_texts:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
            return text_features

        if get_image:
            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(label)
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype),
                                                                                              shared_ctx,
                                                                                              deep_compound_prompts_vision)
            image_features = image_features[:, 0]

            return image_features

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(label)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype),
                                                                                          shared_ctx,
                                                                                          deep_compound_prompts_vision)
        image_features_last = image_features_last[:, 0]
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]

        features_non_proj = self.vision_bottleneck(image_features_non_proj)
        features = self.vision_bottleneck_proj(image_features)

        if self.training:
            cls_score = self.vision_classifier(features_non_proj.float())
            cls_score_proj = self.vision_classifier_proj(features.float())

            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()

            return text_features, image_features, logits, [cls_score, cls_score_proj], [image_features_last,
                                                                                        image_features_non_proj,
                                                                                        image_features]
        else:
            return torch.cat((image_features_non_proj, image_features), dim=1)


def deepspeed_wrapper(model, params, ds_config):
    model_engine, optimizer, _, _ = deepspeed.initialize(args=params, model=model, model_parameters=params,
                                                         config=ds_config)
    return model_engine, optimizer


def train_prompter(model,
                   dataloader,
                   epochs,
                   pretrained=None):
    loss_func = SupConLoss("cuda")

    def train_batch_prompter(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        text_features = model(image, label, get_texts=True)
        image_features = image_features_list[indices]
        loss = loss_func(text_features, image_features, label, label) + \
               loss_func(image_features, text_features, label, label)
        return loss

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("Building custom CLIP")

    with torch.no_grad():
        model.eval()
        index_list = []
        image_features_list = []
        for images, target, cams, seqs, indices in dataloader:
            images = images.cuda()
            target = target.cuda()
            image_features = model(images, target, get_image=True)
            for image_feature, index in zip(image_features, indices):
                image_features_list.append(image_feature.cpu())
                index_list.append(index)
        index_list = torch.stack(index_list, dim=0).cuda()
        image_features_list = torch.stack(image_features_list, dim=0).cuda()
        image_features_list = image_features_list[torch.argsort(index_list)]

    if pretrained is not None:
        load_pretrained_weights(model.prompt_learner, pretrained)
    model.train()

    optimizer = torch.optim.Adam(model.prompt_learner.parameters(), lr=0.00035, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, epochs, 1e-6, 0.00001, 5)

    for epoch in range(epochs):
        iterator = tqdm(dataloader)

        for images, target, cams, seqs in iterator:
            batch = images, target
            optimizer.zero_grad()
            loss = train_batch_prompter(batch)
            optimizer.step()
            iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))
        checkpoint_path = "/".join((params.save_path, "clip_model_prompter_{}.pth".format(epoch)))
        torch.save(model.prompt_learner.state_dict(), checkpoint_path)
        scheduler.step(epoch)
    model.eval()


def train_vision_model(ds_args,
                       model,
                       dataloader,
                       epochs,
                       pretrained=None):
    print("Turning off gradients in both the prompter and the text encoder")
    for name, param in model.named_parameters():
        if "image_encoder" not in name and "vision_classifier" not in name and "vision_bottleneck" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    model, optimizer = deepspeed_wrapper(model, filter(lambda p: p.requires_grad, model.parameters()), ds_args)

    def train_batch_vision_model(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        cls_scores, image_features_list = model(image, label)[3:]
        cls_score1, cls_score2 = cls_scores
        image_features_last, image_features_non_proj, image_features = image_features_list

        loss = 0.25 * F.cross_entropy(cls_score1, label, label_smoothing=0.1) + \
               0.25 * F.cross_entropy(cls_score2, label, label_smoothing=0.1)
        if params.training_mode != "cocoop":
            output = image_features @ text_features.t()
            loss += F.cross_entropy(output, label, label_smoothing=0.1)
        if params.bs >= 4:
            loss += triplet_loss(image_features_last.float(), label) + \
                    triplet_loss(image_features_non_proj.float(), label) + \
                    triplet_loss(image_features.float(), label)
        return loss

    print("Building custom CLIP")

    if params.training_mode != "cocoop":
        with torch.no_grad():
            model.eval()
            text_features = []
            for i in range(n_cls):
                label = torch.tensor([i]).cuda()
                text_feature = model(label=label, get_texts=True)
                text_features.append(text_feature)
            text_features = torch.stack(text_features, dim=0).squeeze().cuda()

    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.image_encoder, pretrained)
        load_pretrained_weights(model.vision_classifier, pretrained)
        load_pretrained_weights(model.vision_classifier_proj, pretrained)
        load_pretrained_weights(model.vision_bottleneck, pretrained)
        load_pretrained_weights(model.vision_bottleneck_proj, pretrained)

    optimizer = torch.optim.Adam(list(model.image_encoder.parameters()) + \
                                 list(model.vision_classifier.parameters()) + \
                                 list(model.vision_classifier_proj.parameters()) + \
                                 list(model.vision_bottleneck.parameters()) + \
                                 list(model.vision_bottleneck_proj.parameters()), lr=0.000005, weight_decay=1e-4)
    scheduler = ConstantWarmupScheduler(optimizer,
                                        torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs // 3, epochs // 3 * 2]),
                                        10,
                                        1e-5)
    triplet_loss = WeightedRegularizedTriplet()

    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)

    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for images, target, cams, seqs, indices in iterator:
            batch = images, target
            loss = train_batch_vision_model(batch)
            model.backward(loss)
            model.step()
            iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))

        ckpt_id = loss.item()
        model.save_checkpoint(params.save_path if os.path.exists(params.save_path) else "clip_model_weight.pth",
                              ckpt_id)

    model.eval()


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs_stage1", default=10, type=int)
    args.add_argument("--epochs_stage2", type=int, default=10)
    args.add_argument("--root", default="./", type=str)
    args.add_argument("--model", default="ViT-B/16", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=1, type=int)
    args.add_argument("--save_path", default="clip_model_prompter.pth")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--local_rank", default=0, type=int)
    args.add_argument("--deepspeed_config", type=str, default="deepspeed_prompter_config.json")
    args.add_argument("--training_mode", type=str, choices=["coop", "cocoop", "maple"])
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    trainer_config = json.load(open(params.deepspeed_config))
    image_height, image_width = params.height, int(params.height * params.ratio)
    url = clip_maple._MODELS[params.model]
    model_path = clip_maple._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if params.training_mode == "maple":
        design_details = {"trainer": 'MaPLe',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": 4}
        model = build_model_maple(state_dict or model.state_dict(), image_height, image_width, design_details)
    elif params.training_mode == "cocoop":
        model = build_model_cocoop(state_dict or model.state_dict(), image_height // 16, image_width // 16, 16)
    elif params.training_mode == "coop":
        model = build_model_coop(state_dict or model.state_dict(), image_height // 16, image_width // 16, 16)
    else:
        raise NotImplementedError

    model = model.cuda()
    loader_train, n_cls = get_loader_train(params.root, params.bs, image_height, image_width,
                                    "vit" if "ViT" in params.model else "rn")

    train_prompter(model,
                   loader_train,
                   params.epochs_stage1)
    train_vision_model(trainer_config,
                       model,
                       loader_train,
                       params.epochs_stage2)
