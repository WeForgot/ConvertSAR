# PNG2SAML

This project documents my ongoing experiments in building a machine learning pipeline that can convert arbitrary PNG images into [Symbol Art Markdown Language (SAML)](https://pso2na.arks-layer.com/) files — the vector-like art format used in *Phantasy Star Online 2*.

The goal: create a model that helps artistically-challenged folks (like me) turn their ideas into clean, layered symbol art usable in-game.

---

## 🔧 What’s in here?

Right now, this repo contains:

- 🧠 A clean, config-driven **CLIP-style pretraining pipeline** using PyTorch
- 🔌 Modular architecture for plug-and-play encoders and decoders
- 💾 Checkpointing, metric logging, and YAML-based experiment tracking
- 📁 Run folders that capture all outputs and configurations per experiment

---

## 🎯 Project Goals

- Train a model that can convert a given PNG image into a corresponding `.saml` file
- Support pretraining (e.g. contrastive CLIP-style) to improve image + symbol representation
- Enable generative decoding using transformer or recurrent architectures
- (Future) Use reinforcement learning to reward more **efficient** SAML generation (fewer layers, same output)

---

## ⚙️ Installation & Dependencies

This project uses:

- Python 3.8+
- PyTorch
- OpenCV (used later for SAML construction & preprocessing)
- torchvision
- tqdm
- einops
- yaml
- lxml
- PIL
- webcolors

```bash
pip install torch torchvision opencv-python tqdm einops pyyaml lxml pillow webcolors
```

---

## 🚀 Training an Experiment

All training scripts are unified under a config-driven pipeline. To run CLIP pretraining:

```bash
python train_clip.py --config=configs/clip/clip_multi_tiny.yaml
```

Your runs will be logged under:
```
runs/
└── clip/
    └── 0/
        ├── config.yaml
        ├── train.csv
        ├── test.csv
        ├── latest.pt
        └── best.pt
```

---

## 📦 Data

**Note:** Training data and model weights are not included. Due to the nature of Symbol Art and its community creators, I cannot legally or ethically distribute the dataset.

You’ll need to bring your own `.saml` files + corresponding `.png` images. Once you have that, you can plug them into the data pipeline and go.

---

## 📌 Roadmap

- [x] CLIP-style image/text encoder setup
- [x] Generative decoding (transformer + recurrent decoder support)
- [ ] Layerwise SAML construction via intermediate model outputs
- [ ] RL fine-tuning based on image similarity and layer count minimization
- [ ] (Optional) Create a community permissible dataset for SAML generator training

---

## 📜 License

MIT License — you can use this code however you want as long as attribution is preserved. If it helps you make wild symbol art? Even better.

---

## 🙏 Acknowledgments

Shoutout to the PSO2 Symbol Art community, the creators of the `.saml` ecosystem, and everyone who ever suffered through making detailed art with grayscale ovals.