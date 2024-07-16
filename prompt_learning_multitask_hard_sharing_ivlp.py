import os
import argparse
import numpy as np
import copy
from itertools import zip_longest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from tqdm import tqdm

from utils import load_pretrained_weights
from data_prepare import get_loader_train, get_loader_train_multitask, get_loader_train_sampled, \
    get_loader_train_sampled_multitask, get_loader
from evaluate import R1_mAP_eval
from schedulers import WarmupMultiStepLR, create_scheduler
from losses import SupConLoss, WeightedRegularizedTriplet, CrossEntropyLabelSmooth, WeightedRegularizedTripletXBM

cudnn.enabled = True
cudnn.deterministic = True

from text_encoder import TextEncoder
from coop import build_model as build_model_coop, \
    PromptLearner as PromptLearnerCoop
from maple import build_model as build_model_maple, VLPromptLearner
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


def get_gauss(mu, sigma, max_epochs):
    gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    gauss_weights = np.array([gauss(a) for a in range(1, max_epochs + 1)])
    gauss_weights = gauss_weights / sum(gauss_weights)
    return gauss_weights


def state_dict_weighting(main_dict, weightage, prompt_only=False):
    # Average all parameters
    updated_dict = copy.deepcopy(main_dict)
    if not prompt_only:
        for key in main_dict:
            updated_dict[key] = main_dict[key] * weightage
        return updated_dict
    else:
        return main_dict * weightage


def state_dict_add(dict1, dict2, prompt_only=False):
    # Average all parameters
    if not prompt_only:
        modified_dict = dict2
        for key in dict1:
            modified_dict[key] = (modified_dict[key] + dict1[key])
        return modified_dict
    else:
        return dict1 + dict2


class XBM:
    def __init__(self, xbm_size, embed_dim):
        self.K = xbm_size
        self.feats = torch.zeros(self.K, embed_dim).cuda() * -1
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda() * -1
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != -1

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size


class Classifier(nn.Module):
    def __init__(self, classnames):
        super().__init__()
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

    def forward(self, image_features_non_proj, image_features):
        features_non_proj = self.vision_bottleneck(image_features_non_proj)
        cls_score = self.vision_classifier(features_non_proj.float())
        features = self.vision_bottleneck_proj(image_features)
        cls_score_proj = self.vision_classifier_proj(features.float())
        return [cls_score, cls_score_proj]


class CustomCLIPIVLP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # self.prompt_learner = VLPromptLearner(classnames, clip_model, params.train_dataset)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image=None, label=None, get_image=False, text_features=None):

        if get_image:
            if params.amp:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(image)
            else:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(
                    image.type(self.dtype))
            image_features = image_features[:, 0]

            return image_features

        if self.training:
            if params.amp:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(image)
            else:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(
                    image.type(self.dtype))
            image_features_last = image_features_last[:, 0]
            image_features_non_proj = image_features_non_proj[:, 0]
            image_features = image_features[:, 0]

            logits = image_features @ text_features.t()

            return text_features, image_features, logits, [image_features_last, image_features_non_proj,
                                                           image_features], image_features
        else:
            if params.amp:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(image)
            else:
                image_features_last, image_features_non_proj, image_features = self.image_encoder(
                    image.type(self.dtype))
            image_features_non_proj = image_features_non_proj[:, 0]
            image_features = image_features[:, 0]
            return torch.cat((image_features_non_proj, image_features), dim=1)


def train_prompter_ivlp(model,
                        model_prompter1,
                        model_prompter2,
                        model_text_encoder1,
                        model_text_encoder2,
                        dataloader_train_val1,
                        dataloader_train_val2,
                        epochs,
                        pretrained=None):
    print("Building custom CLIP")
    if params.amp:
        model = model.float()
        model_prompter1 = model_prompter1.float()
        model_prompter2 = model_prompter2.float()
        model_text_encoder1 = model_text_encoder1.float()
        model_text_encoder2 = model_text_encoder2.float()

    with torch.no_grad():
        num_image1 = num_image2 = 0
        for n_iter, (img, vid, target_cam, target_view, indices) in enumerate(dataloader_train_val1):
            num_image1 += vid.size(0)
        batch = params.bs
        iter1 = num_image1 // batch
        for n_iter, (img, vid, target_cam, target_view, indices) in enumerate(dataloader_train_val2):
            num_image2 += vid.size(0)
        iter2 = num_image2 // batch

    if pretrained is not None:
        load_pretrained_weights(model_prompter1, pretrained)
        load_pretrained_weights(model_prompter2, pretrained)

    print("Turning off gradients in both the image and the text encoder")

    learnable_params = [{"params": model_prompter1.parameters(), "lr": 0.00035, "weight_decay": 1e-4}]
    learnable_params += [{"params": model_prompter2.parameters(), "lr": 0.00035, "weight_decay": 1e-4}]
    for name, param in model.named_parameters():
        if "VPT" in name:
            learnable_params += [{"params": param, "lr": 0.00035, "weight_decay": 1e-5}]

    for name, param in model_text_encoder1.named_parameters():
        if "VPT" in name:
            learnable_params += [{"params": param, "lr": 0.00035, "weight_decay": 1e-4}]

    for name, param in model_text_encoder2.named_parameters():
        if "VPT" in name:
            learnable_params += [{"params": param, "lr": 0.00035, "weight_decay": 1e-4}]

    optimizer = torch.optim.Adam(learnable_params, lr=0.00035, weight_decay=1e-4)
    scheduler = create_scheduler(optimizer, epochs, 1e-6, 0.00001, 5)
    scaler = GradScaler()
    loss_func = SupConLoss("cuda")

    if not os.path.exists(params.save_path):
        os.mkdir(params.save_path)
    saving_path = os.path.join(params.save_path, params.training_mode, params.train_dataset)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    gauss_weights = get_gauss(mu=60, sigma=45, max_epochs=params.epochs_stage1)
    previous_model_gpa = None

    for epoch in range(1, epochs + 1):
        scheduler.step(epoch)
        model.train()
        model_prompter1.train()
        model_prompter2.train()
        model_text_encoder1.train()
        model_text_encoder2.train()

        i = j = 0
        cnt = 0
        dataloader_train_val_iter1 = copy.copy(dataloader_train_val1)
        dataloader_train_val_iter2 = copy.copy(dataloader_train_val2)
        dataloader_train_val_iter1 = iter(dataloader_train_val_iter1)
        dataloader_train_val_iter2 = iter(dataloader_train_val_iter2)
        while i <= iter1 and j <= iter2:
            optimizer.zero_grad()

            if j > iter2 or cnt == 0:
                cnt ^= 1

                (img, vid, target_cam, target_view, indices) = next(dataloader_train_val_iter1)
                img = img.cuda()
                target = vid.cuda()
                i += 1

                if target.size(0) > 0:
                    with autocast(enabled=True):
                        prompts = model_prompter1(target)
                        tokenized_prompts = model_prompter1.tokenized_prompts
                        text_features = model_text_encoder1(prompts, tokenized_prompts)
                        image_features = model(img, target, get_image=True)
                    loss_i2t = loss_func(image_features, text_features, target, target)
                    loss_t2i = loss_func(text_features, image_features, target, target)

                    loss = loss_i2t + loss_t2i

                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()

                    torch.cuda.synchronize()
                    if (i + 1) % 100 == 0:
                        print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                              .format(epoch, (i + 1), len(dataloader_train_val1),
                                      loss, scheduler._get_lr(epoch)[0]))
            else:
                cnt ^= 1
                (img, vid, target_cam, target_view, indices) = next(dataloader_train_val_iter2)
                img = img.cuda()
                target = vid.cuda()
                j += 1

                if target.size(0) > 0:
                    with autocast(enabled=True):
                        prompts = model_prompter2(target)
                        tokenized_prompts = model_prompter2.tokenized_prompts
                        text_features = model_text_encoder2(prompts, tokenized_prompts)
                        image_features = model(img, target, get_image=True)
                    loss_i2t = loss_func(image_features, text_features, target, target)
                    loss_t2i = loss_func(text_features, image_features, target, target)

                    loss = loss_i2t + loss_t2i

                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()

                    torch.cuda.synchronize()
                    if (j + 1) % 100 == 0:
                        print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                              .format(epoch, (j + 1), len(dataloader_train_val2),
                                      loss, scheduler._get_lr(epoch)[0]))

        current_epoch_gauss_weights = gauss_weights[epoch - 1]
        if previous_model_gpa is None:
            previous_model_gpa = state_dict_weighting(copy.deepcopy(model.state_dict()), current_epoch_gauss_weights)
        else:
            previous_model_gpa = state_dict_add(
                state_dict_weighting(copy.deepcopy(model.state_dict()), current_epoch_gauss_weights),
                previous_model_gpa)

        if epoch == params.epochs_stage1 - 1:
            model.load_state_dict(previous_model_gpa)

        if epoch % 20 == 0 or epoch == params.epochs_stage1:
            checkpoint_path = "/".join((saving_path, "clip_model_prompter1_{}.pth".format(epoch - 1)))
            torch.save(model_prompter1.state_dict(), checkpoint_path)
            checkpoint_path = "/".join((saving_path, "clip_model_prompter2_{}.pth".format(epoch - 1)))
            torch.save(model_prompter2.state_dict(), checkpoint_path)

    model.eval()


def train_vision_model_ivlp(model,
                            model_prompter1,
                            model_prompter2,
                            model_text_encoder1,
                            model_text_encoder2,
                            classifier1,
                            classifier2,
                            dataloader1,
                            dataloader2,
                            epochs,
                            pretrained=None):
    print("Building custom CLIP")

    with torch.no_grad():
        model_text_encoder1.eval()
        model_text_encoder2.eval()
        text_features1 = []
        text_features2 = []
        for i in range(n_cls1):
            label = torch.tensor([i]).cuda()
            with autocast():
                prompts = model_prompter1(label)
                tokenized_prompts = model_prompter1.tokenized_prompts
                text_feature = model_text_encoder1(prompts, tokenized_prompts)
            text_features1.append(text_feature)
        text_features1 = torch.cat(text_features1, dim=0).cuda()

        for i in range(n_cls2):
            label = torch.tensor([i]).cuda()
            with autocast():
                prompts = model_prompter2(label)
                tokenized_prompts = model_prompter2.tokenized_prompts
                text_feature = model_text_encoder2(prompts, tokenized_prompts)
            text_features2.append(text_feature)
        text_features2 = torch.cat(text_features2, dim=0).cuda()

    model.train()
    classifier1.train()
    classifier2.train()

    if pretrained is not None:
        load_pretrained_weights(model.image_encoder, pretrained)
        load_pretrained_weights(model.vision_classifier, pretrained)
        load_pretrained_weights(model.vision_classifier_proj, pretrained)
        load_pretrained_weights(model.vision_bottleneck, pretrained)
        load_pretrained_weights(model.vision_bottleneck_proj, pretrained)

    print("Turning off gradients in both the prompter and the text encoder")
    base_lr = 5e-6
    learnable_params = []

    for name, param in prompter1.named_parameters():
        param.requires_grad_(False)
    for name, param in prompter2.named_parameters():
        param.requires_grad_(False)
    for name, param in model_text_encoder1.named_parameters():
        param.requires_grad_(False)
    for name, param in model_text_encoder2.named_parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        # if "text_encoder" in name or "prompt_learner" in name:
        # experimental
        if "VPT" in name:
            param.requires_grad_(False)
        elif not param.requires_grad:
            continue
        else:
            param.requires_grad_(True)
            if "bias" in name:
                lr = base_lr * 2
                learnable_params += [{"params": [param], "lr": lr, "weight_decay": 1e-5}]
            else:
                learnable_params += [{"params": [param], "lr": base_lr, "weight_decay": 1e-5}]

    for name, param in classifier1.named_parameters():
        if not param.requires_grad:
            continue
        else:
            if "bias" in name:
                lr = base_lr * 2
                learnable_params += [{"params": [param], "lr": lr, "weight_decay": 1e-4}]
            else:
                learnable_params += [{"params": [param], "lr": base_lr, "weight_decay": 1e-4}]

    for name, param in classifier2.named_parameters():
        if not param.requires_grad:
            continue
        else:
            if "bias" in name:
                lr = base_lr * 2
                learnable_params += [{"params": [param], "lr": lr, "weight_decay": 1e-4}]
            else:
                learnable_params += [{"params": [param], "lr": base_lr, "weight_decay": 1e-4}]

    xbm1 = XBM(2 * params.bs, 512)
    xbm2 = XBM(2 * params.bs, 512)

    optimizer = torch.optim.Adam(learnable_params, lr=base_lr, weight_decay=1e-4)
    scheduler = WarmupMultiStepLR(optimizer, [30, 50], 0.1, 0.1, 10)
    scaler = GradScaler()
    triplet_loss = WeightedRegularizedTriplet(0.3)
    triplet_loss_xbm = WeightedRegularizedTripletXBM(0.3)
    ce_loss1 = CrossEntropyLabelSmooth(n_cls1)
    ce_loss2 = CrossEntropyLabelSmooth(n_cls2)

    saving_path = os.path.join(params.save_path, params.training_mode, params.train_dataset)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    gauss_weights = get_gauss(mu=30, sigma=30, max_epochs=params.epochs_stage2)
    previous_model_gpa = None

    for epoch in range(epochs):
        scheduler.step()
        loss_sum = 0.
        cnt = 0

        for data1, data2 in zip(dataloader1, dataloader2):
            if data1:
                batch = data1[:2]
                with autocast():
                    image, label = batch[:2]
                    image = image.cuda()
                    label = label.cuda()
                    loss = 0.
                    prompts = model_prompter1(label)
                    tokenized_prompts = model_prompter1.tokenized_prompts
                    image_features_list, image_features_proj = model(image, text_features=text_features1[label])[3:]
                    image_features_last, image_features_non_proj, image_features = image_features_list

                    # with torch.no_grad():
                    #     zero_shot_features_non_proj, zero_shot_features = ZS_image_encoder(image)[1:]
                    #     zero_shot_features = zero_shot_features[:, 0]
                    #     zero_shot_features_non_proj = zero_shot_features_non_proj[:, 0]
                    # loss += F.smooth_l1_loss(torch.cat((image_features_non_proj, image_features_proj), dim=1), torch.cat((zero_shot_features_non_proj, zero_shot_features), dim=1), reduction="mean")

                    cls_score = classifier1(image_features_non_proj, image_features)
                    cls_score1, cls_score2 = cls_score

                    loss += 0.25 * ce_loss1(cls_score1, label) + \
                            0.25 * ce_loss1(cls_score2, label)
                    output = image_features_proj @ text_features1.t()
                    loss += ce_loss1(output, label)

                    if epoch >= 10:
                        xbm1.enqueue_dequeue(image_features.detach(), label.detach())
                        image_features_xbm, label_xbm = xbm1.get()

                        loss += triplet_loss(image_features_last, label) + \
                                triplet_loss(image_features_non_proj, label) + \
                                triplet_loss(image_features, label) + \
                                0.2 * triplet_loss_xbm(image_features, label, image_features_xbm, label_xbm)
                    else:
                        loss += triplet_loss(image_features_last, label) + \
                                triplet_loss(image_features_non_proj, label) + \
                                triplet_loss(image_features, label)
                loss_sum += loss
                cnt += 1
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if data2:
                batch = data2[:2]
                with autocast():
                    image, label = batch[:2]
                    image = image.cuda()
                    label = label.cuda()
                    loss = 0.
                    prompts = model_prompter2(label)
                    tokenized_prompts = model_prompter2.tokenized_prompts
                    image_features_list, image_features_proj = model(image, text_features=text_features2[label])[3:]
                    image_features_last, image_features_non_proj, image_features = image_features_list

                    # with torch.no_grad():
                    #     zero_shot_features_non_proj, zero_shot_features = ZS_image_encoder(image)[1:]
                    #     zero_shot_features = zero_shot_features[:, 0]
                    #     zero_shot_features_non_proj = zero_shot_features_non_proj[:, 0]
                    # loss += F.smooth_l1_loss(torch.cat((image_features_non_proj, image_features_proj), dim=1), torch.cat((zero_shot_features_non_proj, zero_shot_features), dim=1), reduction="mean")

                    cls_score = classifier2(image_features_non_proj, image_features)
                    cls_score1, cls_score2 = cls_score

                    loss += 0.25 * ce_loss2(cls_score1, label) + \
                            0.25 * ce_loss2(cls_score2, label)
                    output = image_features_proj @ text_features2.t()
                    loss += ce_loss2(output, label)
                    if epoch >= 10:
                        xbm2.enqueue_dequeue(image_features.detach(), label.detach())
                        image_features_xbm, label_xbm = xbm2.get()
                        loss += triplet_loss(image_features_last, label) + \
                                triplet_loss(image_features_non_proj, label) + \
                                triplet_loss(image_features, label) + \
                                0.2 * triplet_loss_xbm(image_features, label, image_features_xbm, label_xbm)
                    else:
                        loss += triplet_loss(image_features_last, label) + \
                                triplet_loss(image_features_non_proj, label) + \
                                triplet_loss(image_features, label)
                loss_sum += loss
                cnt += 1
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        print("epoch: {}, loss avg: {}".format(epoch, loss_sum / cnt))

        current_epoch_gauss_weights = gauss_weights[epoch]
        if previous_model_gpa is None:
            previous_model_gpa = state_dict_weighting(copy.deepcopy(model.state_dict()), current_epoch_gauss_weights)
        else:
            previous_model_gpa = state_dict_add(
                state_dict_weighting(copy.deepcopy(model.state_dict()), current_epoch_gauss_weights),
                previous_model_gpa)

        if epoch == params.epochs_stage2 - 1:
            model.load_state_dict(previous_model_gpa)

        if epoch % 20 == 0 or epoch == params.epochs_stage2 - 1:
            checkpoint_path = "/".join((saving_path, "clip_model_weight_{}.pth".format(epoch)))
            torch.save(model.state_dict(), checkpoint_path)

    model.eval()


def test_prompter(model,
                  clip_weight,
                  loader_test):
    # model = model.float()
    model.eval()
    if clip_weight is not None:
        load_pretrained_weights(model, clip_weight)

    embeddings = []
    targets = []
    camera_ids = []
    sequence_ids = []

    with torch.no_grad():
        for i, (images, target, cams, seqs, indices) in enumerate(tqdm(loader_test)):
            images = images.cuda()
            image_features_merged = model(images)

            embeddings.append(image_features_merged)
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
    gallery_embeddings = gallery_embeddings.cpu()
    query_embeddings = query_embeddings.cpu()
    evaluator = R1_mAP_eval(len(query_labels), max_rank=10, feat_norm=True)
    evaluator.reset()
    evaluator.update((torch.cat((query_embeddings, gallery_embeddings), dim=0),
                      torch.cat((query_labels, gallery_labels), dim=0),
                      torch.cat((query_cams, gallery_cams), dim=0)))
    cmc, mAP = evaluator.compute()
    return cmc, mAP


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs_stage1", default=10, type=int)
    args.add_argument("--epochs_stage2", default=60, type=int)
    args.add_argument("--root", default="./", type=str)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=1, type=int)
    args.add_argument("--save_path", default="./checkpoints")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--height_multitask", default=224, type=int)
    args.add_argument("--ratio_multitask", default=0.5, type=float)
    args.add_argument("--amp", action="store_true")
    args.add_argument("--training_mode", type=str, default="ivlp", choices=["ivlp"])
    args.add_argument("--train_dataset", type=str, default="market1501",
                      choices=["market1501", "dukemtmc", "personx", "msmt17", "veri", "vehicleid"])
    args.add_argument("--train_dataset_multitask", type=str, default="dukemtmc",
                      choices=["market1501", "dukemtmc", "personx", "msmt17", "veri", "vehicleid"])
    args.add_argument("--test_dataset", type=str, default="dukemtmc",
                      choices=["market1501", "dukemtmc", "msmt17", "veri", "vehicleid"])
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    image_height, image_width = params.height, int(params.height * params.ratio)
    image_height_multitask, image_width_multitask = params.height_multitask, int(
        params.height_multitask * params.ratio_multitask)
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
                          "language_depth": 12,
                          "vision_ctx": 2,
                          "language_ctx": 2}
        model_zero = build_model_coop(state_dict or model.state_dict(), image_height // 16, image_width // 16,
                                      16)  # build_model_maple(state_dict or model.state_dict(), image_height, image_width, design_details_zero) # one thing is the stride size, should we keep the original size?
        model = build_model_maple(state_dict or model.state_dict(), image_height, image_width, design_details)

    else:
        raise NotImplementedError

    model = model.cuda()

    _, loader_train_val1, n_cls1, car_types_train = get_loader_train(params.root, params.bs, image_height, image_width,
                                                                     "vit" if "ViT" in params.model else "rn", True,
                                                                     params.train_dataset)
    loader_train_sampled1, _ = get_loader_train_sampled(params.root, params.bs, image_height, image_width,
                                                        "vit" if "ViT" in params.model else "rn", params.train_dataset)

    _, loader_train_val2, n_cls2, car_types_train = get_loader_train(params.root, params.bs, image_height_multitask,
                                                                     image_width_multitask,
                                                                     "vit" if "ViT" in params.model else "rn", True,
                                                                     params.train_dataset_multitask)
    loader_train_sampled2, _ = get_loader_train_sampled(params.root, params.bs, image_height_multitask,
                                                        image_width_multitask,
                                                        "vit" if "ViT" in params.model else "rn",
                                                        params.train_dataset_multitask)
    if params.training_mode == "ivlp":
        state_dict = torch.load("./clip_imagenet_pretrained_ivlp.pth.tar-5")["state_dict"]
        with torch.no_grad():
            ZS_image_encoder = model_zero.visual.cuda()
        # reset prompt learner and positional embedding
        from collections import OrderedDict

        state_dict_reseted = OrderedDict()
        state_dict_textual = OrderedDict()
        for layer in state_dict:
            if "VPT" in layer:
                state_dict_reseted[layer] = state_dict[layer]
            if "transformer.resblocks" in layer and "VPT" in layer:
                state_dict_textual[layer.lstrip("text_encoder.")] = state_dict[layer]
        prompter1 = VLPromptLearner(n_cls1, model, params.train_dataset).cuda()
        prompter2 = VLPromptLearner(n_cls2, model, params.train_dataset_multitask).cuda()
        text_encoder1 = TextEncoder(model).cuda()
        text_encoder2 = TextEncoder(copy.deepcopy(model)).cuda()
        model = CustomCLIPIVLP(model).cuda()
        text_encoder1.load_state_dict(state_dict_textual, strict=False)
        text_encoder2.load_state_dict(state_dict_textual, strict=False)
        model.load_state_dict(state_dict_reseted, strict=False)
    else:
        raise NotImplementedError

    loader_gallery, loader_query, loader_gallery_augmented, loader_query_augmented = get_loader(params.root, params.bs,
                                                                                                image_height,
                                                                                                image_width,
                                                                                                "vit" if "ViT" in params.model else "rn",
                                                                                                params.test_dataset)

    classifier1 = Classifier(n_cls1).cuda()
    classifier2 = Classifier(n_cls2).cuda()

    train_prompter_ivlp(model,
                        prompter1,
                        prompter2,
                        text_encoder1,
                        text_encoder2,
                        loader_train_val1,
                        loader_train_val2,
                        params.epochs_stage1)

    # if params.amp:
    #     model = model.float()
    #     model_prompter1 = prompter1.float()
    #     model_prompter2 = prompter2.float()

    # prompter1.load_state_dict(torch.load(os.path.join(params.save_path, params.training_mode, params.train_dataset, "clip_model_prompter1_{}.pth".format(params.epochs_stage1-1))))
    # prompter2.load_state_dict(torch.load(os.path.join(params.save_path, params.training_mode, params.train_dataset, "clip_model_prompter2_{}.pth".format(params.epochs_stage1-1))))

    train_vision_model_ivlp(model,
                            prompter1,
                            prompter2,
                            text_encoder1,
                            text_encoder2,
                            classifier1,
                            classifier2,
                            loader_train_sampled1,
                            loader_train_sampled2,
                            params.epochs_stage2)
    latest_model = "/".join((os.path.join(params.save_path, params.training_mode, params.train_dataset),
                             "clip_model_weight_{}.pth".format(params.epochs_stage2 - 1)))
    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = \
        test_prompter(model, latest_model, loader_gallery)
    embeddings_query, targets_query, cameras_query, sequences_query = \
        test_prompter(model, latest_model, loader_query)
    embeddings_gallery_augmented, _, _, _ = \
        test_prompter(model, latest_model, loader_gallery_augmented)
    embeddings_query_augmented, _, _, _ = \
        test_prompter(model, latest_model, loader_query_augmented)
    embeddings_gallery = (embeddings_gallery + embeddings_gallery_augmented) / 2
    embeddings_query = (embeddings_query + embeddings_query_augmented) / 2
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
