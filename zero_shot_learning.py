import argparse
from collections import OrderedDict

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import model_adaptor
from evaluate import R1_mAP_eval
from data_prepare import get_prompts, get_loader, get_prompts_augmented


def load_model(model_name, classnames, templates, weights=None):
    model, _ = clip.load(model_name)
    model.eval()
    '''
    # text encoder not trained
    if weights is not None:
    weights = torch.load(weights)
    matched_weights = OrderedDict()
    for key in weights:
        if key.startswith("text_encoder"):
            matched_key = ".".join(key.split(".")[1:])
            matched_weights[matched_key] = weights[key].to(model.state_dict()[matched_key].dtype)
    model.load_state_dict(matched_weights, strict=False)
    '''

    def zeroshot_classifier(classnames, templates: dict):
        with torch.no_grad():
            if params.augmented_template:
                class_embeddings = []
                for classname in classnames:
                    texts = templates[classname]
                    texts = clip.tokenize(texts).cuda()
                    class_embedding = model.encode_text(texts)  # embed with text encoder
                    class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
                    class_embedding = class_embedding.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    class_embeddings.append(class_embedding)
                class_embeddings = torch.stack(class_embeddings, dim=0).cuda()
            else:
                texts = [templates[classname] for classname in classnames]
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        return class_embeddings

    zeroshot_weights = zeroshot_classifier(classnames, templates)
    return zeroshot_weights, model


def inference(model,
              bottleneck,
              bottleneck_proj,
              zeroshot_weights,
              loader,
              loader_augment,
              multimodal):
    model.eval()
    bottleneck.eval()
    bottleneck_proj.eval()

    embeddings = []
    embeddings_proj = []
    targets = []
    camera_ids = []
    sequence_ids = []

    with torch.no_grad():
        for i, (images, target, cams, seqs) in enumerate(tqdm(loader)):
            images = images.cuda()

            # predict
            image_features_last, image_features, image_features_proj = model.encode_image(images)
            image_features = image_features[:, 0]
            image_features_proj = image_features_proj[:, 0]
            # image_features = bottleneck(image_features)
            # image_features_proj = bottleneck_proj(image_features_proj)
            logits = torch.cat((image_features, image_features_proj), dim=1)

            if multimodal:
                embeddings_proj.append(image_features_proj)
                logits = image_features
            embeddings.append(logits)
            targets.append(target)
            camera_ids.append(cams)
            sequence_ids.append(seqs)

        for i, (images, target, cams, seqs) in enumerate(tqdm(loader_augment)):
            images = images.cuda()

            # predict
            image_features_last, image_features, image_features_proj = model.encode_image(images)
            image_features = image_features[:, 0]
            image_features_proj = image_features_proj[:, 0]
            # image_features = bottleneck(image_features)
            # image_features_proj = bottleneck_proj(image_features_proj)
            if multimodal:
                image_features_proj = (embeddings_proj[i] + image_features_proj) / 2.
                image_features_proj /= image_features_proj.norm(dim=-1, keepdim=True)
                logits = 1./0.07 * image_features_proj.float() @ zeroshot_weights.T.float()
                logits = logits.softmax(dim=-1)
                image_features = (embeddings[i] + image_features) / 2.
                logits = torch.cat((image_features.float(), logits), dim=1)
            else:
                image_features = torch.cat((image_features, image_features_proj), dim=1)
                image_features = (embeddings[i] + image_features) / 2.
                logits = image_features.float()

            embeddings[i] = logits

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
    evaluator = R1_mAP_eval(len(query_labels), max_rank=50, feat_norm=True)
    evaluator.reset()
    evaluator.update((torch.cat((query_embeddings, gallery_embeddings), dim=0),
                      torch.cat((query_labels, gallery_labels), dim=0),
                      torch.cat((query_cams, gallery_cams), dim=0)))
    cmc, mAP = evaluator.compute()
    return cmc, mAP


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--root", default="", type=str)
    args.add_argument("--bs", default=64, type=int)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--augmented_template", action="store_true")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    args.add_argument("--mm", action="store_true")
    args.add_argument("--clip_weights", type=str, default="Market1501_clipreid_ViT-B-16_60.pth")
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    model_name = params.model
    image_height, image_width = params.height, int(params.height * params.ratio)
    loader_gallery, loader_query, loader_gallery_augmented, loader_query_augmented = get_loader(params.root, params.bs,
                                                                                                image_height,
                                                                                                image_width)
    if params.augmented_template:
        identity_list, template_dict = get_prompts_augmented("Market-1501_Attribute/market_attribute.mat")
    else:
        identity_list, template_dict = get_prompts("Market-1501_Attribute/market_attribute.mat")
    zeroshot_weights, model = load_model(model_name, identity_list, template_dict)
    model, bottleneck, bottleneck_proj = model_adaptor(model, image_height, image_width, params.clip_weights)

    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = inference(model, bottleneck, bottleneck_proj, zeroshot_weights, loader_gallery, loader_gallery_augmented, params.mm)
    embeddings_query, targets_query, cameras_query, sequences_query = inference(model, bottleneck, bottleneck_proj, zeroshot_weights, loader_query, loader_query_augmented, params.mm)
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
