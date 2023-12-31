import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from tqdm import tqdm

from utils import load_pretrained_weights, model_adaptor
from data_prepare import get_loader_train, get_loader
from evaluate import R1_mAP_eval
from schedulers import ConstantWarmupScheduler, create_scheduler
from losses import SupConLoss, WeightedRegularizedTriplet

cudnn.enabled = True
cudnn.deterministic = True


from cocoop import PromptLearner, TextEncoder
from maple import build_model, MultiModalPromptLearner, TextEncoder as TextEncoderMaple
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


class CustomCLIPCoCoop(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, params)
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

    def forward(self, image, label):
        _, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]

        if self.training:
            image_features_non_proj = self.vision_bottleneck(image_features_non_proj)
            cls_score = self.vision_classifier(image_features_non_proj.float())
            image_features = self.vision_bottleneck_proj(image_features)
            cls_score_proj = self.vision_classifier_proj(image_features.float())

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            prompts = self.prompt_learner(image_features, label)
            tokenized_prompts = self.tokenized_prompts
            logit_scale = self.logit_scale.exp()
            logits = []
            texts = []

            for pts_i, imf_i in zip(prompts, image_features):
                text_features = self.text_encoder(pts_i, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                l_i = logit_scale * imf_i @ text_features.t()
                logits.append(l_i)
                texts.append(text_features)

            logits = torch.stack(logits)
            texts = torch.stack(texts)

            return texts, image_features, logits, cls_score, cls_score_proj
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

    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(label)
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features_last, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        image_features_last = image_features_last[:, 0]
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]

        features_non_proj = self.vision_bottleneck(image_features_non_proj)
        features = self.vision_bottleneck_proj(image_features)

        if self.training:
            cls_score = self.vision_classifier(features_non_proj.float())
            cls_score_proj = self.vision_classifier_proj(features.float())

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ text_features.t()

            return text_features, image_features, logits, [cls_score, cls_score_proj], [image_features_last, image_features_non_proj, image_features]
        else:
            return torch.cat((image_features_non_proj, image_features), dim=1)


def train_prompter(model,
                   dataloader,
                   epochs,
                   pretrained=None):

    def train_batch_prompter(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        text_features, image_features = model(image, label)[:2]
        loss = loss_func(text_features, image_features, label, label) + \
               loss_func(image_features, text_features, label, label)
        return loss

    print("Building custom CLIP")
    if params.amp:
        model = model.float()
    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.prompt_learner, pretrained)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    # optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.001)
    # scheduler = ConstantWarmupScheduler(optimizer,
    #                                     torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs),
    #                                     5,
    #                                     1e-6)
    optimizer = torch.optim.Adam(model.prompt_learner.parameters(), lr=0.00035, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, epochs, 1e-6, 0.00001, 5)
    scaler = GradScaler()
    loss_func = SupConLoss("cuda")

    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)

    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        if params.amp:
            for images, target, cams, seqs in iterator:
                batch = images, target
                with autocast():
                    loss = train_batch_prompter(batch)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))
        else:
            for images, target, cams, seqs in iterator:
                batch = images, target
                loss = train_batch_prompter(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))

        scheduler.step(epoch)
        checkpoint_path = "/".join((params.save_path, "clip_model_prompter_{}.pth".format(epoch)))
        torch.save(model.prompt_learner.state_dict(), checkpoint_path)

    model.eval()


def train_vision_model(model,
                       dataloader,
                       epochs,
                       pretrained=None):

    def train_batch_vision_model(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        output, cls_scores, image_features_list = model(image, label)[1:]
        cls_score1, cls_score2 = cls_scores
        image_features_last, image_features_non_proj, image_features = image_features_list
        loss = F.cross_entropy(output, label) + \
               0.25 * F.cross_entropy(cls_score1, label, label_smoothing=0.1) + \
               0.25 * F.cross_entropy(cls_score2, label, label_smoothing=0.1)
        if params.bs >= 4:
            loss += triplet_loss(image_features_last, label) + \
                triplet_loss(image_features_non_proj, label) + \
                triplet_loss(image_features, label)
        return loss

    print("Building custom CLIP")
    if params.amp:
        model = model.float()
    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.image_encoder, pretrained)
        load_pretrained_weights(model.vision_classifier, pretrained)
        load_pretrained_weights(model.vision_classifier_proj, pretrained)
        load_pretrained_weights(model.bottleneck, pretrained)
        load_pretrained_weights(model.bottleneck_proj, pretrained)

    print("Turning off gradients in both the prompter and the text encoder")
    for name, param in model.named_parameters():
        if "image_encoder" not in name and "vision_classifier" not in name and "vision_bottleneck" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    optimizer = torch.optim.Adam(list(model.image_encoder.parameters()) + \
                                 list(model.vision_classifier.parameters()) + \
                                 list(model.vision_classifier_proj.parameters()) + \
                                 list(model.bottleneck.parameters()) + \
                                 list(model.bottleneck_proj.parameters()), lr=0.000005, weight_decay=1e-4)
    scheduler = ConstantWarmupScheduler(optimizer,
                                        torch.optim.lr_scheduler.MultiStepLR(optimizer, [epochs // 3, epochs // 3 * 2]),
                                        10,
                                        1e-5)
    scaler = GradScaler()
    triplet_loss = WeightedRegularizedTriplet()

    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)

    for epoch in range(epochs):
        iterator = tqdm(dataloader)
        if params.amp:
            for images, target, cams, seqs in iterator:
                batch = images, target
                with autocast():
                    loss = train_batch_vision_model(batch)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))
        else:
            for images, target, cams, seqs in iterator:
                batch = images, target
                loss = train_batch_vision_model(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iterator.set_description("epoch: {}, loss: {}".format(epoch, loss))

        scheduler.step()
        checkpoint_path = "/".join((params.save_path, "clip_model_image_encoder_{}.pth".format(epoch)))
        torch.save(model.image_encoder.state_dict(), checkpoint_path)

    model.eval()


def test_prompter(model,
                  prompt_learner_weight,
                  loader_test):
    model.eval()
    load_pretrained_weights(model.prompt_learner, prompt_learner_weight)

    embeddings = []
    targets = []
    camera_ids = []
    sequence_ids = []

    with torch.no_grad():
        for i, (images, target, cams, seqs) in enumerate(tqdm(loader_test)):
            images = images.cuda()
            logits = model(images)

            embeddings.append(logits)
            targets.append(target)
            camera_ids.append(cams)
            sequence_ids.append(seqs)
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)
    camera_ids = torch.cat(camera_ids, dim=0)
    sequence_ids = torch.cat(sequence_ids, dim=0)
    return embeddings, targets, camera_ids, sequence_ids


def get_cmc_map(
        gallery_embeddings,
        query_embeddings,
        gallery_labels,
        query_labels,
        gallery_cams,
        query_cams
):
    gallery_embeddings = gallery_embeddings.cpu().float()
    query_embeddings = query_embeddings.cpu().float()
    evaluator = R1_mAP_eval(len(query_labels), max_rank=50, feat_norm=True)
    evaluator.reset()
    evaluator.update((torch.cat((query_embeddings, gallery_embeddings), dim=0),
                      torch.cat((query_labels, gallery_labels), dim=0),
                      torch.cat((query_cams, gallery_cams), dim=0)))
    cmc, mAP = evaluator.compute()
    return cmc, mAP


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs_stage1", default=10, type=int)
    args.add_argument("--epochs_stage2", default=10, type=int)
    args.add_argument("--root", default="./", type=str)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=1, type=int)
    args.add_argument("--save_path", default="./checkpoints")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--amp", action="store_true")
    args.add_argument("--clip_weights", type=str, default="Market1501_clipreid_ViT-B-16_60.pth")
    args.add_argument("--maple", action="store_true")
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    image_height, image_width = params.height, int(params.height * params.ratio)
    if params.maple:
        url = clip_maple._MODELS[params.model]
        model_path = clip_maple._download(url)
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        design_details = {"trainer": 'MaPLe',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "maple_length": 2}
        model = build_model(state_dict or model.state_dict(), image_height, image_width, design_details)
        model = model.cuda()
    else:
        design_details = None
        model, _ = clip.load(params.model)
        model = model_adaptor(model, image_height, image_width, design_details, params.clip_weights)[0]

    model.eval()

    loader_train, n_cls = get_loader_train(params.root, params.bs, image_height, image_width,
                                    "vit" if "ViT" in params.model else "rn")
    if params.maple:
        model = CustomCLIPMaple(n_cls, model).cuda()
    else:
        model = CustomCLIPCoCoop(n_cls, model).cuda()

    loader_gallery, loader_query = get_loader(params.root, params.bs,
                                              image_height,
                                              image_width,
                                              "vit" if "ViT" in params.model else "rn")[:2]

    train_prompter(model,
                   loader_train,
                   params.epochs_stage1)
    train_vision_model(model,
                       loader_train,
                       params.epochs_stage2)
    latest_model = sorted(glob.glob("/".join((params.save_path, "clip_model_image_encoder_*"))), reverse=True)[0]
    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = \
        test_prompter(model, latest_model, loader_gallery)
    embeddings_query, targets_query, cameras_query, sequences_query = \
        test_prompter(model, latest_model, loader_query)
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
