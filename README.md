# Image-Captioning
# Flickr30k Image Captioning

## Description
This project implements an *image captioning model using the Flickr30k dataset. It employs a CNN-based encoder (ResNet-50) to extract spatial image features and an attention-based LSTM decoder to generate descriptive captions. The model is trained to generate human-like captions for images, with performance evaluated using BLEU and METEOR metrics. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Features
- Dataset: Processes Flickr30k images and captions (~31,000 images, 5 captions each).
- Encoder: Uses pre-trained ResNet-50 (fine-tuned on layers 3 and 4) to extract 7x7 feature maps.
- Decoder: Attention-based LSTM for generating captions word-by-word.
- Training: Supports training with Adam optimizer, learning rate scheduling, and gradient clipping.
- Evaluation: Computes BLEU and METEOR scores on validation and test sets.
- Visualization*: Plots training/validation loss, BLEU, and METEOR scores; displays sample test images with generated captions.

## Prerequisites
- Python 3.8+
- Libraries:
  - `torch`
  - `torchvision`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `Pillow`
  - `nltk`
  - `scikit-learn`
- CUDA-enabled GPU (optional, for faster training)
- Flickr30k dataset (images and captions)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flickr30k-image-captioning.git
   cd flickr30k-image-captioning

#Install dependencies
pip install torch torchvision numpy pandas matplotlib Pillow nltk scikit-learn

#Download NLTK data
import nltk
nltk.download('wordnet')

#Download the Flickr30k dataset
  Obtain the dataset from Kaggle or the official source.
  Place the images in archive/flickr30k_images/.
  Place the captions file (captions.txt) in archive/.

#Usage
Ensure the dataset is structured as:

archive/
├── flickr30k_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── captions.txt

#Run the script:
python image_captioning.py

This will:
  Preprocess the dataset.
  Train the model for 10 epochs.
  Evaluate on validation/test sets.
  Save trained models (encoder.pth, decoder.pth).
  Display training curves and sample test images with generated captions.

#Code Structure
  Data Preparation: Loads captions, builds vocabulary, tokenizes text, and splits data into train/val/test sets.
  Dataset: Custom FlickrDataset class with image transformations (resize, crop, normalize).
  Model:
    EncoderCNN: Extracts spatial features using ResNet-50.
    DecoderRNN_Attn: Generates captions with attention mechanism.
  Training: Uses cross-entropy loss, Adam optimizer, and StepLR scheduler.
  Evaluation: Computes BLEU/METEOR scores and generates captions for test images.
  Visualization: Plots metrics and displays captioned images.

#Results
  Training Loss: Decreases over 10 epochs.  
  Validation Metrics:
    BLEU: Measures caption similarity to ground truth.
    METEOR: Evaluates semantic quality.
    Sample Output: Displays 5 test images with generated captions.

#Example
After training, the model might generate captions like:
  Image: A dog running on grass.
  Caption: "A dog runs on a grassy field."

#Notes
Adjust batch_size, num_epochs, or vocab_size in the script for different hardware or performance.
The model fine-tunes only ResNet-50's layers 3 and 4 to balance performance and computation.
Ensure sufficient disk space for the dataset (~10GB for images).

#Contributing
Fork the repository.
Create a feature branch (git checkout -b feature-branch).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-branch).
Open a Pull Request.
