import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm

from utils import load_pretrained_weights, model_adaptor
from data_prepare import get_loader_train


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        ctx_init = "a photo of a"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            print("Initializing class-specific contexts")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def train_batch(model, batch):
    image, label = batch
    image = image.cuda()
    label = label.cuda()
    output = model(image)
    loss = F.cross_entropy(output, label)
    return loss


def train_prompter(classnames,
                   clip_model,
                   dataloader,
                   epochs,
                   pretrained=None):
    print("Building custom CLIP")
    model = CustomCLIP(classnames, clip_model)

    if pretrained is not None:
        load_pretrained_weights(model.prompt_learner, pretrained)

    print("Turning off gradients in both the image and the text encoder")
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        for i, (images, target, cams, seqs) in enumerate(tqdm(dataloader)):
            batch = images, target
            loss = train_batch(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(clip_model.prompt_learner, params.save_path if os.path.exists(params.save_path) else "clip_model_prompter.pt")
    return clip_model


def params_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", default=32, type=int)
    args.add_argument("--root", default="", type=str)
    args.add_argument("--model", default="RN50", choices=clip.available_models(), type=str)
    args.add_argument("--bs", default=64, type=int)
    args.add_argument("--save_path", default="clip_model_prompter.pt")
    args.add_argument("--height", default=224, type=int)
    args.add_argument("--ratio", default=0.5, type=float)
    return args.parse_args()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    params = params_parser()
    model, transforms = clip.load(params.model)
    image_height, image_width = params.height, int(params.height * params.ratio)
    model = model_adaptor(model, image_height, image_width)
    loader_train = get_loader_train(transforms, params.root, params.bs, image_height, image_width)
    classnames = ["person " + str(i) for i in range(751)]

    trained_model = train_prompter(classnames,
                                   model,
                                   loader_train,
                                   params.epochs)
