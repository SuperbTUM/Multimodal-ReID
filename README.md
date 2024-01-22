# Multimodal-ReID

As multimodal models go viral these days, we make an attempt to apply CLIP in traditional ReID tasks.

**Installation**

```bash
pip install -r requirements.txt
```

**Quick Start**

Market1501-pretrained ViT/16 model, 
thanks for the contribution of CLIP-ReID: 
[ViT_Market1501](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view), 
[ViT_DukeMTMC](https://drive.google.com/file/d/1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY/view)

```bash
python3 zero_shot_learning.py --model ViT-B/16 --augmented_template --height 256
```

Train with Prompt Engineering
```bash
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --test_dataset dukemtmc
```
Or
```bash
deepspeed prompt_learning_deepspeed.py --height 256 --bs 64 --epochs_stage1 120 --training_mode ivlp
```

**Cross-domain Evaluation**

M->D

| Method                         | Acc@1 | Acc@5 | Acc@10 | mAP   |
|--------------------------------|-------|-------|--------|-------|
| CoOp                           | 70.8% | 82.1% | 85.7%  | 51.9% |
| CoOp w/. prompt augmentation   | 71.0% | 82.3% | 85.7%  | 52.5% |
| IVLP                           | 70.0% | 82.4% | 85.6%  | 52.4% |
| IVLP w/. text encoder unlocked | 71.2% | 82.5% | 86.2%  | 52.9% |


D->M

| Method                         | Acc@1 | Acc@5 | Acc@10 | mAP   |
|--------------------------------|-------|-------|--------|-------|
| CoOp                           | 71.4% | 84.7% | 89.2%  | 44.6% |
| IVLP w/. text encoder unlocked | 71.9% | 86.1% | 90.0%  | 45.8% |