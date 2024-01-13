# Multimodal-ReID

As multimodal models go viral these days, we make an attempt to apply CLIP in traditional ReID tasks.

**Installation**

```bash
pip install -r requirements.txt
```

**Quick Start**

Market1501-pretrained ViT/16 model, 
thanks for the contribution of CLIP-ReID: 
[Weight_ViT](https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view), 
[Weight_CNN](https://drive.google.com/file/d/1sBqCr5LxKcO9J2V0IvLQPb0wzwVzIZUp/view)

```bash
python3 zero_shot_learning.py --model ViT-B/16 --augmented_template --height 256
```

Train with Prompt Engineering
```bash
python3 prompt_learning.py --model ViT-B/16 --height 256 --bs 64 --amp --epochs_stage1 120 --epochs_stage2 60 --training_mode ivlp  --test_dataset dukemtmc
```
Or
```bash
deepspeed prompt_learning_deepspeed.py --height 256 --amp
```

**Cross-domain Evaluation**

CoOp M->D Rank@1:67.8%, Rank@5:79.4%, Rank@10:83.0%, mAP:46.8%

IVLP M->D Rank@1:70.0%, Rank@5:82.4%, Rank@10:85.6%, mAP:52.4%