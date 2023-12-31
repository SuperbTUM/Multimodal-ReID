import numpy as np
from scipy import io
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from datasets import dataset_market


class ToSquare:
    def __init__(self):
        pass

    def __call__(self, pil_img, background_color=(0, 0, 0)):
        return self.expand2square(pil_img, background_color)

    @staticmethod
    def expand2square(pil_img, background_color=(0, 0, 0)):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class reidDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
        self.to_square = ToSquare()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        detailed_info = list(self.images[item])
        detailed_info[0] = Image.open(detailed_info[0]).convert("RGB")
        # detailed_info[0] = self.to_square.expand2square(detailed_info[0], (255, 255, 255))
        if self.transform:
            detailed_info[0] = self.transform(detailed_info[0])
        detailed_info[1] = torch.tensor(detailed_info[1])
        for i in range(2, len(detailed_info)):
            detailed_info[i] = torch.tensor(detailed_info[i], dtype=torch.long)
        return detailed_info


def get_loader_train(root, batch_size, image_height, image_width, model_type):
    transform_train = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.RandomHorizontalFlip(),
        transforms.Pad((10, 5)),
        transforms.RandomCrop((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5) if model_type == "vit" else (0.485, 0.456, 0.406), std=(0.5, 0.5, 0.5) if model_type == "vit" else (0.229, 0.224, 0.225)),
        transforms.RandomErasing()
    ])
    dataset = dataset_market.Market1501(root="/".join((root, "Market1501")))
    num_pids = dataset.num_train_pids
    reid_dataset_train = reidDataset(dataset.train, transform_train)
    loader_train = DataLoader(reid_dataset_train, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
    return loader_train, num_pids


def get_loader(root, batch_size, image_height, image_width, model_type):
    transform_test = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5) if model_type == "vit" else (0.485, 0.456, 0.406), std=(0.5, 0.5, 0.5) if model_type == "vit" else (0.229, 0.224, 0.225)),
    ])
    preprocess = transform_test
    transform_test_augmented = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.RandomHorizontalFlip(1.0),
        # transforms.Pad((int(5 * ratio), 5)),
        # transforms.RandomCrop((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5) if model_type == "vit" else (0.485, 0.456, 0.406), std=(0.5, 0.5, 0.5) if model_type == "vit" else (0.229, 0.224, 0.225)),
    ])
    preprocess_augmented = transform_test_augmented
    dataset = dataset_market.Market1501(root="/".join((root, "Market1501")))
    reid_dataset_gallery = reidDataset(dataset.gallery, preprocess)
    reid_dataset_query = reidDataset(dataset.query, preprocess)
    loader_gallery = DataLoader(reid_dataset_gallery, batch_size=batch_size, num_workers=4, shuffle=False,
                                pin_memory=True)
    loader_query = DataLoader(reid_dataset_query, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
    reid_dataset_gallery_augmented = reidDataset(dataset.gallery, preprocess_augmented)
    reid_dataset_query_augmented = reidDataset(dataset.query, preprocess_augmented)
    loader_gallery_augmented = DataLoader(reid_dataset_gallery_augmented, batch_size=batch_size, num_workers=4,
                                          shuffle=False, pin_memory=True)
    loader_query_augmented = DataLoader(reid_dataset_query_augmented, batch_size=batch_size, num_workers=4,
                                        shuffle=False, pin_memory=True)
    return loader_gallery, loader_query, loader_gallery_augmented, loader_query_augmented


def get_prompts_simple(identity_list, num_class):
    sentence_templates = ["itap of a {}", "a bad photo of the {}", "a origami {}", "a photo of the large {}",
                          "a {} in a video game", "art of the {}", "a photo of the small {}"]
    templates = [list() for _ in range(num_class)]
    for i in range(num_class):
        for st in sentence_templates:
            templates[i].append(st.format("person no." + str(i)))
    return identity_list, {identity: template for identity, template in zip(identity_list, templates)}


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
    color_mapping_lower = {0: "black", 1: "white", 2: "pink", 3: "purple", 4: "yellow", 5: "gray", 6: "blue",
                           7: "green", 8: "brown"}

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
        template_basic = "a {age} {gender} person no.{index} with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}, ".format(
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
        if template_advanced == "carrying ":
            template_advanced = ""
            template_hat = template_hat.rstrip(", ")
        template_advanced = template_advanced.rstrip(", ")
        return template_basic + template_hat + template_advanced + "."

    index = 0
    for age, backpack, bag, handbag, lower_body_clothing, length_lower_body, sleeve, hair_length, hat, gender in zip(
            *attributes):
        template = get_prompt(gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag,
                              handbag, age, index)
        index += 1
        templates.append(template)
    return identity_list, {identity: template for identity, template in zip(identity_list, templates)}


def get_prompts_augmented(file_name):
    sentence_templates = ["itap of a {}", "a bad photo of the {}", "a origami {}", "a photo of the large {}",
                          "a {} in a video game", "art of the {}", "a photo of the small {}"]
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
    color_mapping_lower = {0: "black", 1: "white", 2: "pink", 3: "purple", 4: "yellow", 5: "gray", 6: "blue",
                           7: "green", 8: "brown"}

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
        template_basic1 = "{age} {gender} person no.{index} on my left or right side with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}".format(
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
        template_basic2 = "{age} {gender} person no.{index} walking with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}".format(
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
        template_basic3 = "{age} {gender} person no.{index} rushing with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}".format(
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
        template_basic4 = "{age} {gender} person no.{index} in the distance with {hair_length}, {color1} {sleeve}, {color2} {length_lower_body} {lower_body_clothing}".format(
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
        template_hat = "wearing nothing on head" if hat == 1 else "wearing a hat"
        template_advanced = "carrying "
        items = []
        if backpack != 1:
            items.append("a backpack")
        if bag != 1:
            items.append("a bag")
        if handbag != 1:
            items.append("a handbag")
        if items:
            if len(items) > 1:
                items = " and ".join([", ".join(items[:-1]), items[-1]])
            else:
                items = items[0]
            template_advanced += items
        else:
            template_advanced += "nothing"
        templates = [", ".join((template_basic1, template_hat, template_advanced)),
                     ", ".join((template_basic2, template_hat, template_advanced)),
                     ", ".join((template_basic3, template_hat, template_advanced)),
                     ", ".join((template_basic4, template_hat, template_advanced)),
                     ", ".join((template_basic1, template_advanced, template_hat)),
                     ", ".join((template_basic2, template_advanced, template_hat)),
                     ", ".join((template_basic3, template_advanced, template_hat)),
                     ", ".join((template_basic4, template_advanced, template_hat))
                     ]
        ensembled_templates = []
        for sentence_template in sentence_templates:
            for tlt in templates:
                ensembled_templates.append(sentence_template.format(tlt))
        return ensembled_templates

    index = 0
    for age, backpack, bag, handbag, lower_body_clothing, length_lower_body, sleeve, hair_length, hat, gender in zip(
            *attributes):
        template = get_prompt(gender, hair_length, sleeve, length_lower_body, lower_body_clothing, hat, backpack, bag,
                              handbag, age, index)
        index += 1
        templates.append(template)
    return identity_list, {identity: template for identity, template in zip(identity_list, templates)}
