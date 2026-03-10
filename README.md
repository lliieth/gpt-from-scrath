# CyberSec GPT Project

This project implements a small GPT-style language model using PyTorch.

The model is trained on cybersecurity related text to understand and generate security concepts.

## Topics included
- Malware
- Phishing
- Ransomware
- Vulnerabilities
- Exploits
- Encryption
- Network security

## Model Idea
The model learns using **Next Token Prediction**, which is the core training objective used in Large Language Models (LLMs).

## Files
- transformer.py → transformer architecture
- attention.py → self-attention implementation
- pretrain.py → model pretraining
- finetune.py → fine tuning on cybersecurity data
- CyberSec_GPT_Project.ipynb → Colab notebook demo

## Goal
To demonstrate how a simplified GPT model can be built from scratch and adapted for cybersecurity knowledge.
