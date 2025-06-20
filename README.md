# 🛰️ Remote Sensing Image Captioning for Satellite Images

This repository contains an implementation of remote sensing image captioning for satellite imagery. The goal is to evaluate the **robustness** of different encoder-decoder architectures in generating accurate and meaningful captions for satellite images.

## 🧠 Project Overview

The task involves training deep learning models to generate textual descriptions (captions) for satellite images. This is particularly useful for automating image understanding in remote sensing applications.

### 🗂️ Dataset

- **RSICD** (Remote Sensing Image Captioning Dataset)
- Each image is associated with **5 human-annotated captions**

## 🏗️ Model Architectures

We experimented with **two encoders** and **two decoder configurations**:

### 🔹 Encoders
- **ReNeSt-50** (pretrained on ImageNet-1K)
- **MobileNetV2** (pretrained on ImageNet)

### 🔸 Decoders
- **Single-layer LSTM**
- **4-layer stacked LSTM**

## ⚙️ Training Details

- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** Adam  
- **Tokenizer:** Custom tokenizer built for the RSICD dataset  
- **Training Objective:** Minimize caption generation error on RSICD dataset

- ## ⚙️ Evaluation
- ## Evaluated using 4 metrics: Bleu, Rouge 1, Rouge L, Bert score F1

## 🧪 Objective

The main goal of this work is to:
> **Assess the robustness and performance of different encoder-decoder combinations for generating captions from satellite imagery.**

This includes understanding:
- The impact of different encoder backbones
- The effect of deeper decoder architectures (1-layer vs. 4-layer LSTM)
