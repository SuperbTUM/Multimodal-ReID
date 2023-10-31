# Multimodal-ReID

As multimodal models go viral these days, we make an attempt to apply CLIP in traditional ReID tasks.

Installation

```bash
pip install git+https://github.com/openai/CLIP.git
```

Quick Start

```bash
python3 zero_shot_learning.py --model ViT-B/32 --augmented_template
```

Train with Prompt Engineering
```bash
python3 prompt_learning.py
```