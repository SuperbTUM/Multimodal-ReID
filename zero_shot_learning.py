import argparse

import clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import model_adaptor
from evaluate import evaluate
from data_prepare import get_prompts, get_loader, get_prompts_augmented


def load_model(model_name, classnames, templates):
    model, preprocess = clip.load(model_name)

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
    return zeroshot_weights, preprocess, model


def inference(model, zeroshot_weights, loader, loader_augment):
    embeddings = []
    targets = []
    camera_ids = []
    sequence_ids = []

    with torch.no_grad():
        for i, (images, target, cams, seqs) in enumerate(tqdm(loader)):
            images = images.cuda()
            # target = target.cuda()

            # predict
            image_features = model.encode_image(images)
            logits = image_features

            embeddings.append(logits)
            targets.append(target)
            camera_ids.append(cams)
            sequence_ids.append(seqs)

        for i, (images, target, cams, seqs) in enumerate(tqdm(loader_augment)):
            images = images.cuda()
            # target = target.cuda()

            # predict
            image_features = model.encode_image(images)
            image_features = (embeddings[i] + image_features) / 2.
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights.T
            logits = logits.softmax(dim=-1)
            logits = F.normalize(logits, dim=-1)

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
    CMC = torch.IntTensor(gallery_embeddings.size(0)).zero_()
    ap = 0.0
    for i in range(query_embeddings.size(0)):
        ap_tmp, CMC_tmp = evaluate(query_embeddings[i],
                                   query_labels[i],
                                   query_cams[i],
                                   gallery_embeddings,
                                   gallery_labels,
                                   gallery_cams)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / query_embeddings.size(0)  # average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / query_embeddings.size(0)))


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--root", default="", type=str)
    args.add_argument("--bs", default=64, type=int)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--augmented_template", action="store_true")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    model_name = params.model
    if params.augmented_template:
        identity_list, template_dict = get_prompts_augmented("Market-1501_Attribute/market_attribute.mat")
    else:
        identity_list, template_dict = get_prompts("Market-1501_Attribute/market_attribute.mat")
    zeroshot_weights, transforms_, model = load_model(model_name, identity_list, template_dict)
    image_height, image_width = params.height, int(params.height * params.ratio)
    model = model_adaptor(model, image_height, image_width)

    loader_gallery, loader_query, loader_gallery_augmented, loader_query_augmented = get_loader(transforms_, params.root, params.bs, image_height, image_width)

    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = inference(model, zeroshot_weights, loader_gallery, loader_gallery_augmented)
    embeddings_query, targets_query, cameras_query, sequences_query = inference(model, zeroshot_weights, loader_query, loader_query_augmented)
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
