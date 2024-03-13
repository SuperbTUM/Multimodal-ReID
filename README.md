# Multimodal-ReID

As multimodal models go viral these days, we make an attempt to apply CLIP in traditional ReID tasks.

**Installation**

```bash
pip install -r requirements.txt
```

**Quick Start**

ImageNet-pretrained ViT/16 CoOp model, 
thanks for the contribution of CLIP-ReID: 
[ViT_Market1501](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view), 
[ViT_DukeMTMC](https://drive.google.com/file/d/1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY/view)

ImageNet-Pretrained ViT/16 IVLP model,
thanks for the contribution of multimodal-prompt-learning:
[IVLP_Clip](https://drive.google.com/file/d/1B7BOjQSzISWVxfeNkEM4qHOGeOCuksaJ/view?usp=sharing)

```bash
python3 zero_shot_learning.py --model ViT-B/16 --augmented_template --height 256 --mm --clip_weights xxx
```

Training Examples with Prompt Engineering
```bash
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --test_dataset dukemtmc --vpt_ctx 4
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --train_dataset dukemtmc --test_dataset market1501 --vpt_ctx 2
```
Or
```bash
deepspeed prompt_learning_deepspeed.py --height 256 --bs 64 --epochs_stage1 120 --training_mode ivlp
```

**Cross-domain Evaluation**

Experimental logs [Only visible to myself atm](https://docs.google.com/document/d/1wBPoy53pGGp1bkmO97LpaA_eDzA4s0OPX8hgvXit8E4/edit?usp=sharing)

Abbreviations: M: Market1501; D: DukeMTMC, MS: MSMT17_V2

M->D

| Method                             | Acc@1     | Acc@5     | Acc@10    | mAP       |
|------------------------------------|-----------|-----------|-----------|-----------|
| CoOp                               | 69.9%     | 81.5%     | 85.4%     | 51.2%     |
| CoOp Enhanced                      | 70.9%     | 82.4%     | 86.1%     | 52.5%     |
| IVLP w/. text encoder unlocked     | 71.3%     | 82.9%     | 85.9%     | 53.0%     |
| ~~+ w/. masked language modeling~~ | ~~71.3%~~ | ~~83.0%~~ | ~~86.4%~~ | ~~52.5%~~ |


D->M

| Method                             | Acc@1     | Acc@5     | Acc@10    | mAP       |
|------------------------------------|-----------|-----------|-----------|-----------|
| CoOp                               | 71.4%     | 84.7%     | 89.2%     | 44.6%     |
| CoOp Enhanced                      | 73.0%     | 86.1%     | 89.7%     | 46.3%     |
| IVLP                               | 73.1%     | 85.8%     | 90.1%     | 46.5%     |
| ~~+ w/. masked language modeling~~ | ~~73.6%~~ | ~~85.4%~~ | ~~89.9%~~ | ~~46.1%~~ |


MS->M

| Method                         | Acc@1 | Acc@5 | Acc@10 | mAP   |
|--------------------------------|-------|-------|--------|-------|
| CoOp w/. tricks                | 74.2% | 86.6% | 91.0%  | 49.3% |
| CoOp Enhanced                  | 75.6% | 88.1% | 91.9%  | 50.7% |
| IVLP w/. text encoder unlocked | 75.4% | 87.7% | 91.3%  | 50.0% |

**In-domain Evaluation**

Veri776

| Method                                     | Acc@1 | Acc@5 | Acc@10 | mAP   |
|--------------------------------------------|-------|-------|--------|-------|
| CoOp (256 x 256)                           | 97.4% | 98.6% | 99.3%  | 83.3% |
| IVLP w/. text encoder unlocked (224 x 224) | 97.6% | 98.9% | 99.5%  | 83.7% |

DukeMTMC

| Method  | Acc@1 | Acc@5 | Acc@10 | mAP   |
|---------|-------|-------|--------|-------|
| CoOp    | 90.4% | 96.0% | 97.6%  | 83.3% |
| IVLP    | 91.0% | 96.2% | 97.5%  | 83.7% |