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
            age,
            index
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
        template_basic = "{age} {gender} person {index} with {hair_length}, {sleeve}, {length_lower_body} {lower_body_clothing}, ".format(
                       age=age,
                       gender=gender,
                       hair_length=hair_length,
                       sleeve=sleeve,
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