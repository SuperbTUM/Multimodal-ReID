# Multimodal-ReID

As multimodal models go viral these days, we make an attempt to apply CLIP variants in traditional ReID tasks.

This work has been accepted by [ISVC](https://www.isvc.net/) 24' and will be published on [Advances in Visual Computing](https://link.springer.com/conference/isvc) by the end of 2024.

[Slides](https://drive.google.com/file/d/156Mi89qjdezJEjDvI6fqkOv1ZzhFdUJt/view?usp=sharing) are now available.

**Installation**

```bash
pip install -r requirements.txt
```

**Quick Start**

This repo does not include concrete prompt generation by GPT-4o(mini).

ImageNet-Pretrained ViT/16 IVLP model,
thanks for the contribution of multimodal-prompt-learning:
[IVLP](https://drive.google.com/file/d/1B7BOjQSzISWVxfeNkEM4qHOGeOCuksaJ/view?usp=sharing)

```bash
python3 zero_shot_learning.py --model ViT-B/16 --augmented_template --height 256 --mm --clip_weights xxx
```

Training Examples with Prompt Engineering
```bash
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --test_dataset dukemtmc
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --train_dataset dukemtmc --test_dataset market1501 --vpt_ctx 2
```
