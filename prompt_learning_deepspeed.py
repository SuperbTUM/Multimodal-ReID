import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler
import deepspeed
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from tqdm import tqdm

from utils import load_pretrained_weights
from data_prepare import get_loader_train, get_loader_train_sampled
from losses import SupConLoss, WeightedRegularizedTriplet
from schedulers import WarmupMultiStepLR, create_scheduler

cudnn.enabled = True
cudnn.deterministic = True

from text_encoder import TextEncoder, TextEncoderAugmented
from coop import build_model as build_model_coop, PromptLearner as PromptLearnerCoop
from maple import build_model as build_model_maple, VLPromptLearner, TextEncoder as TextEncoderMaple
import clip_custom


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

        if self.training:
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            prompts = self.prompt_learner(label)
            tokenized_prompts = self.tokenized_prompts
            logit_scale = self.logit_scale.exp()

            text_features = self.text_encoder(prompts, tokenized_prompts)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()
            return text_features, image_features, logits, [cls_score, cls_score_proj], [image_features_last,
                                                                                        image_features_non_proj,
                                                                                        image_features], image_features
        else:
            return torch.cat((image_features_non_proj, image_features), dim=1)


class CustomCLIPIVLP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(classnames, clip_model)
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
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            return text_features

        if get_image:
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features[:, 0]

            return image_features

        if self.training:
            prompts = self.prompt_learner(label)
            text_features = self.text_encoder(prompts, tokenized_prompts)
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
            image_features_last = image_features_last[:, 0]
            image_features_non_proj = image_features_non_proj[:, 0]
            image_features = image_features[:, 0]

            features_non_proj = self.vision_bottleneck(image_features_non_proj)
            features = self.vision_bottleneck_proj(image_features)
            cls_score = self.vision_classifier(features_non_proj.float())
            cls_score_proj = self.vision_classifier_proj(features.float())

            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.t()

            return text_features, image_features, logits, [cls_score, cls_score_proj], [image_features_last,
                                                                                        image_features_non_proj,
                                                                                        image_features], image_features
        else:
            image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
            image_features_non_proj = image_features_non_proj[:, 0]
            image_features = image_features[:, 0]
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

    # def train_batch_prompter(batch):
    #     image, label = batch[:2]
    #     image = image.cuda()
    #     label = label.cuda()
    #     text_features = model(image, label, get_texts=True)
    #     image_features = image_features_list[indices]
    #     loss = loss_func(text_features, image_features, label, label) + \
    #            loss_func(image_features, text_features, label, label)
    #     return loss

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    print("Building custom CLIP")

    # with torch.no_grad():
    #     model.eval()
    #     index_list = []
    #     image_features_list = []
    #     for images, target, cams, seqs, indices in dataloader:
    #         images = images.cuda()
    #         target = target.cuda()
    #         if params.amp:
    #             with autocast():
    #                 image_features = model(images, target, get_image=True)
    #         else:
    #             image_features = model(images, target, get_image=True)
    #         for image_feature, index in zip(image_features, indices):
    #             image_features_list.append(image_feature.cpu())
    #             index_list.append(index)
    #     index_list = torch.stack(index_list, dim=0).cuda()
    #     image_features_list = torch.stack(image_features_list, dim=0).cuda()
    #     image_features_list = image_features_list[torch.argsort(index_list)]
    labels = []
    image_features = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view, indices) in enumerate(dataloader):
            img = img.cuda()
            target = vid.cuda()
            with autocast():
                image_feature = model(img, target, get_image=True)
            for i, img_feat in zip(target, image_feature):
                labels.append(i)
                image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0).cuda()  # N
        image_features_list = torch.stack(image_features, dim=0).cuda()
        batch = params.bs
        num_image = labels_list.shape[0]
        i_ter = num_image // batch

    if pretrained is not None:
        load_pretrained_weights(model.prompt_learner, pretrained)
    model.train()

    optimizer = torch.optim.Adam(model.prompt_learner.parameters(), lr=0.00035, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, epochs, 1e-6, 0.00001, 5)
    scaler = GradScaler()

    saving_path = os.path.join(params.save_path, params.training_mode, params.train_dataset)

    for epoch in range(1, epochs + 1):
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).cuda()
        for i in range(i_ter + 1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with autocast(enabled=True):
                text_features = model(label=target, get_texts=True)
            loss_i2t = loss_func(image_features, text_features, target, target)
            loss_t2i = loss_func(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            if (i + 1) % 100 == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                      .format(epoch, (i + 1), len(dataloader),
                              loss, scheduler._get_lr(epoch)[0]))

        if epoch % 10 == 0 or epoch == params.epochs_stage1:
            checkpoint_path = "/".join((saving_path, "clip_model_prompter_{}.pth".format(epoch - 1)))
            torch.save(model.prompt_learner.state_dict(), checkpoint_path)
    model.eval()


def train_vision_model(model,
                       dataloader,
                       epochs,
                       pretrained=None):
    print("Turning off gradients in both the prompter and the text encoder")
    for name, param in model.named_parameters():
        # if "text_encoder" in name or "prompt_learner" in name:
        # experimental
        if "prompt_learner" in name:
            param.requires_grad_(False)
        elif not param.requires_grad:
            continue
        else:
            param.requires_grad_(True)

    deepspeed_prompter_config = \
        {
            "train_batch_size": params.bs,
            "gradient_accumulation_steps": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.000005,
                    "weight_decay": 0.0001
                }
            },
            "fp16": {
                "enabled": True,
                "auto_cast": True
            },
            "zero_optimization": {
                "stage": 3,
                "reduce_bucket_size": 5e8,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            }
        }

    model, optimizer = deepspeed_wrapper(model, filter(lambda p: p.requires_grad, model.parameters()),
                                         deepspeed_prompter_config)

    def train_batch_vision_model(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        cls_scores, image_features_list, image_features_proj = model(image, label)[3:]
        cls_score1, cls_score2 = cls_scores
        image_features_last, image_features_non_proj, image_features = image_features_list

        loss = 0.25 * F.cross_entropy(cls_score1, label, label_smoothing=0.1) + \
               0.25 * F.cross_entropy(cls_score2, label, label_smoothing=0.1)
        if params.training_mode != "cocoop":
            output = image_features_proj @ text_features.t()
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
            text_features = torch.cat(text_features, dim=0).cuda()

    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.image_encoder, pretrained)
        load_pretrained_weights(model.vision_classifier, pretrained)
        load_pretrained_weights(model.vision_classifier_proj, pretrained)
        load_pretrained_weights(model.vision_bottleneck, pretrained)
        load_pretrained_weights(model.vision_bottleneck_proj, pretrained)

    # scheduler = WarmupMultiStepLR(optimizer, [30, 50], 0.1, 0.1, 10)
    triplet_loss = WeightedRegularizedTriplet(0.3)

    saving_path = os.path.join(params.save_path, params.training_mode, params.train_dataset)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        for images, target, cams, seqs, indices in iterator:
            batch = images, target
            loss = train_batch_vision_model(batch)
            model.backward(loss)
            model.step()
            iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))

        ckpt_id = loss.item()
        if epoch % 10 == 0 or epoch == params.epochs_stage2 - 1:
            model.save_checkpoint(os.path.join(saving_path, "clip_model_weight.pth"), ckpt_id)

    model.eval()


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs_stage1", default=10, type=int)
    args.add_argument("--epochs_stage2", default=60, type=int)
    args.add_argument("--root", default="./", type=str)
    args.add_argument("--model", default="ViT-B/16", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=1, type=int)
    args.add_argument("--save_path", default="./checkpoints_deepspeed")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--local_rank", default=0, type=int)
    args.add_argument("--training_mode", type=str, choices=["coop", "ivlp"])
    args.add_argument("--train_dataset", type=str, default="market1501", choices=["market1501", "dukemtmc"])
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    image_height, image_width = params.height, int(params.height * params.ratio)
    _, loader_train_val, n_cls = get_loader_train(params.root, params.bs, image_height, image_width,
                                                  "vit" if "ViT" in params.model else "rn",
                                                  True, params.train_dataset)
    loader_train_sampled, _ = get_loader_train_sampled(params.root, params.bs, image_height, image_width,
                                                       "vit" if "ViT" in params.model else "rn", params.train_dataset)

    url = clip_custom._MODELS[params.model]
    model_path = clip_custom._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if params.training_mode == "ivlp":
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 12,
                          "language_depth": 12, "vision_ctx": 4,
                          "language_ctx": 4}
        model = build_model_maple(state_dict or model.state_dict(), image_height, image_width, design_details)
        model = CustomCLIPIVLP(n_cls, model).cuda()
    elif params.training_mode == "coop":
        model = build_model_coop(state_dict or model.state_dict(), image_height // 16, image_width // 16, 16)
        model = CustomCLIPCoop(n_cls, model).cuda()
    else:
        raise NotImplementedError

    model = model.cuda()

    train_prompter(model,
                   loader_train_val,
                   params.epochs_stage1)
    train_vision_model(model,
                       loader_train_sampled,
                       params.epochs_stage2)
