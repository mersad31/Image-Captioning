# 🖼️ Image Captioning on Flickr30k

A PyTorch image-captioning model: a ResNet-50 encoder looks at the picture and an attention-based LSTM decoder writes a sentence describing it.

---

## 🎯 The Problem

Turn an image into a natural-language description, and measure how close the generated caption is to human-written ones.

## 💡 The Approach

| Part | Details |
|---|---|
| Dataset | Flickr30k (~31,000 images, 5 captions each) |
| Encoder | Pretrained ResNet-50 (layers 3 and 4 fine-tuned) giving a 7×7 grid of 49 regions, each projected to 256 dimensions |
| Decoder | Attention + LSTM cell (hidden size 512, attention size 256), generating one word at a time |
| Vocabulary | Top 10,000 words |
| Training | Cross-entropy, Adam (lr = 1e-4), StepLR (step 5, gamma 0.1), gradient clipping, 10 epochs, batch size 16 |
| Evaluation | BLEU and METEOR on validation and test sets |

<!-- TODO: add 3-5 sample images with generated captions, e.g. ![Samples](docs/samples.png) -->

## 📊 Results

The script plots training/validation loss and BLEU/METEOR scores, and shows sample test images with generated captions.

<!-- TODO: add the final BLEU / METEOR table -->

## 🚀 Getting Started

```bash
git clone https://github.com/mersad31/Image-Captioning.git
cd Image-Captioning
pip install torch torchvision numpy pandas matplotlib pillow nltk scikit-learn
python -c "import nltk; nltk.download('wordnet')"
```

Download Flickr30k from Kaggle and arrange it like this (the paths are set in `NLP-project.py`):

```text
archive/
├── flickr30k_images/
│   ├── image1.jpg
│   └── ...
└── captions.txt
```

Run training and evaluation (a CUDA GPU is recommended):

```bash
python NLP-project.py
```

The trained weights are saved as `encoder.pth` and `decoder.pth`.

## 📂 Repository Layout

| File | Purpose |
|---|---|
| `NLP-project.py` | Data preparation, model, training, evaluation, visualization |
| `LICENSE` | License |

## 📝 License

See [LICENSE](LICENSE).
