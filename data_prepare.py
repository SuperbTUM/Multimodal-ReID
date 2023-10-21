import numpy as np
from scipy import io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import dataset_market


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


def get_loader(preprocess, root):
    dataset = dataset_market.Market1501(root="/".join((root, "Market1501")))
    reid_dataset_train = reidDataset(dataset.train, preprocess)
    reid_dataset_gallery = reidDataset(dataset.gallery, preprocess)
    reid_dataset_query = reidDataset(dataset.query, preprocess)
    loader_train = DataLoader(reid_dataset_train, batch_size=64, num_workers=4, shuffle=True, pin_memory=True)
    loader_gallery = DataLoader(reid_dataset_gallery, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    loader_query = DataLoader(reid_dataset_query, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)
    return loader_gallery, loader_query, loader_train


def get_prompts(file_name):
    mat = io.loadmat(file_name)["market_attribute"][0][0]
    mat = mat[0][0][0]
    identity_list = list(map(lambda x: x.item(), mat[-1][0]))
    templates = []
    attributes = []
    upper_colors = []
    lower_colors = []
    for i in range(10):
        attributes.append(mat[i][0])
    for i in range(10, 18):
        upper_colors.append(mat[i][0])
    for i in range(18, 27):
        lower_colors.append(mat[i][0])
    upper_colors = np.array(upper_colors)
    lower_colors = np.array(lower_colors)

    color_mapping_upper = {0: "black", 1: "white", 2: "red", 3: "purple", 4: "yellow", 5: "gray", 6: "blue", 7: "green"}
    color_mapping_lower = {0: "black", 1: "white", 2: "pink", 3: "purple", 4: "yellow", 5: "gray", 6: "blue", 7: "green", 8: "brown"}

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
            age,
            index
    ):
        gender = "male" if gender == 1 else "female"
        hair_length = "short hair" if hair_length == 1 else "long hair"
        sleeve = "long sleeve" if sleeve == 1 else "short sleeve"
        length_lower_body = "long" if length_lower_body == 1 else "short"
        lower_body_clothing = "dress" if lower_body_clothing == 1 else "pants"
        color1_identity, color2_identity = upper_colors[:, index], lower_colors[:, index]
        color1, color2 = "other", "other"

        for i in range(len(color1_identity)):
            if color1_identity[i] != 1:
                color1 = color_mapping_upper[i]
                break

        for i in range(len(color2_identity)):
            if color2_identity[i] != 1:
                color2 = color_mapping_lower[i]
                break

        if age == 1:
            age = "young"
        elif age == 2:
            age = "teenager"
        elif age == 3:
            age = "adult"
        else:
            age = "old"
        template_basic = "a {age} {gender} person {index} with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}, ".format(
                       age=age,
                       gender=gender,
                       hair_length=hair_length,
                       color1=color1,
                       sleeve=sleeve,
                       color2=color2,
                       length_lower_body=length_lower_body,
                       lower_body_clothing=lower_body_clothing,
                       index=index,
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

    index = 0
    for gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag, handbag, age in zip(*attributes):
        template = get_prompt(gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag, handbag, age, index)
        index += 1
        templates.append(template)
    return identity_list, {identity: template for identity, template in zip(identity_list, templates)}