from scipy import io
from tqdm import tqdm
import argparse
from PIL import Image

import clip
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import dataset_market
from evaluate import evaluate


class reidDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        for i in range(2, len(detailed_info)):
            detailed_info[i] = torch.tensor(detailed_info[i], dtype=torch.long)
        return detailed_info


def get_loader(preprocess):
    dataset = dataset_market.Market1501(root="/".join((params.root, "Market1501")))
    reid_dataset_gallery = reidDataset(dataset.gallery, preprocess)
    reid_dataset_query = reidDataset(dataset.query, preprocess)
    loader_gallery = DataLoader(reid_dataset_gallery, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    loader_query = DataLoader(reid_dataset_query, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    return loader_gallery, loader_query


def get_prompts(file_name):
    mat = io.loadmat(file_name)["market_attribute"][0][0]
    mat = mat[0][0][0]
    identity_list = list(map(lambda x: x.item(), mat[-1][0]))
    templates = []
    attributes = []
    for i in range(10):
        attributes.append(mat[i][0])

    def get_prompt(
            gender,
            hair_length,
            sleeve,
            length_lower_body,
            lower_body_clothing,
            hat,
            backpack,
            bag,
            handbag,
            age
    ):
        gender = "male" if gender == 1 else "female"
        hair_length = "short hair" if hair_length == 1 else "long hair"
        sleeve = "long sleeve" if sleeve == 1 else "short sleeve"
        length_lower_body = "long" if length_lower_body == 1 else "short"
        lower_body_clothing = "dress" if lower_body_clothing == 1 else "pants"
        # hat = "no hat" if hat == 1 else "hat"
        # backpack = "no backpack" if backpack == 1 else "backpack"
        # bag = "no bag" if bag == 1 else "bag"
        # handbag = "no handbag" if handbag == 1 else "handbag"
        if age == 1:
            age = "young"
        elif age == 2:
            age = "teenager"
        elif age == 3:
            age = "adult"
        else:
            age = "old"
        template_basic = "A photo of {age} {gender} with {hair_length}, {sleeve}, {length_lower_body} {lower_body_clothing}, ".format(
                       age=age,
                       gender=gender,
                       hair_length=hair_length,
                       sleeve=sleeve,
                       length_lower_body=length_lower_body,
                       lower_body_clothing=lower_body_clothing,
                   )
        template_hat = "" if hat == 1 else "wearing a hat, "
        template_advanced = "carrying "
        if backpack != 1:
            template_advanced += "a backpack, "
        if bag != 1:
            template_advanced += "a bag, "
        if handbag != 1:
            template_advanced += "a handbag, "
        template_advanced = template_advanced.rstrip(", ")
        return template_basic + template_hat + template_advanced + "."

    for gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag, handbag, age in zip(*attributes):
        template = get_prompt(gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag, handbag, age)
        templates.append(template)
    return identity_list, {identity: template for identity, template in zip(identity_list, templates)}


def load_model(model_name, classnames, templates):
    model, preprocess = clip.load(model_name)

    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            texts = [templates[classname] for classname in classnames]
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        return class_embeddings

    zeroshot_weights = zeroshot_classifier(classnames, templates)
    return zeroshot_weights, preprocess, model


def inference(model, zeroshot_weights, loader):
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
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100. * image_features @ zeroshot_weights.T

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
    args.add_argument("--root", default="")
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    return args.parse_args()


if __name__ == "__main__":
    params = params_parser()
    model_name = params.model
    identity_list, template_dict = get_prompts("Market-1501_Attribute/market_attribute.mat")
    zeroshot_weights, transforms, model = load_model(model_name, identity_list, template_dict)
    loader_gallery, loader_query = get_loader(transforms)
    embeddings_gallery, targets_gallery, cameras_gallery, sequences_gallery = inference(model, zeroshot_weights, loader_gallery)
    embeddings_query, targets_query, cameras_query, sequences_query = inference(model, zeroshot_weights, loader_query)
    get_cmc_map(embeddings_gallery, embeddings_query, targets_gallery, targets_query, cameras_gallery, cameras_query)
