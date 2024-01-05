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
from schedulers import ConstantWarmupScheduler
from losses import SupConLoss

cudnn.enabled = True
cudnn.deterministic = True


from cocoop import PromptLearner, TextEncoder


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
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

        self.vision_classifier = nn.Linear(768, classnames, bias=False)
        self.vision_classifier.apply(weights_init_classifier)
        self.vision_classifier_proj = nn.Linear(512, classnames, bias=False)
        self.vision_classifier_proj.apply(weights_init_classifier)

    def forward(self, image, label):
        _, image_features_non_proj, image_features = self.image_encoder(image.type(self.dtype))
        image_features_non_proj = image_features_non_proj[:, 0]
        image_features = image_features[:, 0]
        cls_score = self.vision_classifier(image_features_non_proj.float())
        cls_score_proj = self.vision_classifier_proj(image_features.float())

        if self.training:
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


def train_prompter(classnames,
                   clip_model,
                   dataloader,
                   epochs,
                   pretrained=None):

    def train_batch_prompter(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        text_features, image_features = model(image, label)[:2]
        loss = loss_func(text_features, image_features, label, label) + loss_func(image_features, text_features, label, label)
        return loss

    print("Building custom CLIP")
    if params.amp:
        clip_model = clip_model.float()
    model = CustomCLIPCoCoop(classnames, clip_model).cuda()
    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.prompt_learner, pretrained)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.001)
    scheduler = ConstantWarmupScheduler(optimizer,
                                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs),
                                        1,
                                        1e-5)
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

        scheduler.step()
        checkpoint_path = "/".join((params.save_path, "clip_model_prompter_{}.pth".format(epoch)))
        torch.save(model.prompt_learner.state_dict(), checkpoint_path)

    model.eval()

    return model


def train_vision_model(classnames,
                       clip_model,
                       dataloader,
                       epochs,
                       pretrained=None):

    def train_batch_vision_model(batch):
        image, label = batch[:2]
        image = image.cuda()
        label = label.cuda()
        output, cls_score1, cls_score2 = model(image, label)[2:]
        loss = F.cross_entropy(output, label) + F.cross_entropy(cls_score1, label,
                                                                label_smoothing=0.1) + F.cross_entropy(cls_score2,
                                                                                                       label,
                                                                                                       label_smoothing=0.1)
        return loss

    print("Building custom CLIP")
    if params.amp:
        clip_model = clip_model.float()
    model = CustomCLIPCoCoop(classnames, clip_model).cuda()
    model.train()

    if pretrained is not None:
        load_pretrained_weights(model.image_encoder, pretrained)
        load_pretrained_weights(model.vision_classifier, pretrained)
        load_pretrained_weights(model.vision_classifier_proj, pretrained)

    print("Turning off gradients in both the prompter and the text encoder")
    for name, param in model.named_parameters():
        if "image_encoder" not in name and "vision_classifier" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    optimizer = torch.optim.SGD(list(model.image_encoder.parameters()) + list(model.vision_classifier.parameters()) + list(model.vision_classifier_proj.parameters()), lr=0.001)
    scheduler = ConstantWarmupScheduler(optimizer,
                                        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs),
                                        1,
                                        1e-5)
    scaler = GradScaler()

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

    return model


def test_prompter(clip_model,
                  prompt_learner_weight,
                  loader_test):
    model = CustomCLIPCoCoop(n_cls, clip_model).cuda()
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
    args.add_argument("--epochs", default=10, type=int)
    args.add_argument("--root", default="./", type=str)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=1, type=int)
    args.add_argument("--save_path", default="./checkpoints")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--amp", action="store_true")
    args.add_argument("--clip_weights", type=str, default="Market1501_clipreid_ViT-B-16_60.pth")
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    model, _ = clip.load(params.model)
    model.eval()
    image_height, image_width = params.height, int(params.height * params.ratio)
    model = model_adaptor(model, image_height, image_width, params.clip_weights)[0]
    loader_train, n_cls = get_loader_train(params.root, params.bs, image_height, image_width,
                                    "vit" if "ViT" in params.model else "rn")
    loader_gallery, loader_query = get_loader(params.root, params.bs,
                                              image_height,
                                              image_width,
                                              "vit" if "ViT" in params.model else "rn")[:2]

    trained_model = train_prompter(n_cls,
                                   model,
                                   loader_train,
                                   params.epochs)
    trained_model = train_vision_model(n_cls,
                                       trained_model,
                                       loader_train,
                                       params.epochs)
    latest_model = sorted(glob.glob("/".join((params.save_path, "clip_model_image_encoder_*"))), reverse=True)[0]
    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = \
        test_prompter(trained_model, latest_model, loader_gallery)
    embeddings_query, targets_query, cameras_query, sequences_query = \
        test_prompter(trained_model, latest_model, loader_query)
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
