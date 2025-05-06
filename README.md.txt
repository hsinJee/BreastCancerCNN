
# Breast Cancer Detection in Histopathology Images

## Overview

This project presents a comparative study on classifying breast cancer histopathology images using multiple CNN architectures. A lightweight Convolutional Neural Network (CNN) was built entirely from scratch using NumPy and benchmarked against two powerful pretrained models: **VGG16** and **ResNet50**.

The main focus was on:
- Achieving >90% recall on malignant cases
- Minimizing model size and inference time
- Demonstrating effectiveness of a CPU-trained NumPy model in low-resource settings

---

## Dataset

**BreakHis 200x** histopathology dataset  
- 2013 images (623 benign, 1390 malignant)  
- RGB format, resized to 224x224  
- Source: [BreakHis Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

---

## Architectures

### Custom CNN (NumPy)
- Implemented from scratch
- Batched input support
- Batch normalization
- Adam optimizer
- 3 Conv layers + Dense

### Pretrained Models
- VGG16 (14.7M parameters)
- ResNet50 (23.6M parameters)
- Transfer learning on last 20 layers

---

## Results

| Model        | Accuracy | F1-Score | Recall (Malignant) | Params     | Inference Time |
|--------------|----------|----------|---------------------|------------|----------------|
| Custom CNN   | 84.16%   | 88.73%   | **90.65%**           | 319,106    | **0.31s**      |
| VGG16        | 93.56%   | 95.41%   | 97.12%              | 14.7M      | 1.75s          |
| ResNet50     | 96.04%   | **97.08%** | 95.68%              | 23.6M      | 1.85s          |

---

## Folder Structure

