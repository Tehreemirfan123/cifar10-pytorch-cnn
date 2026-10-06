# CIFAR-10 Image Classification with PyTorch CNN

An end-to-end Machine Learning pipeline utilizing native PyTorch to build, train, evaluate, and serialize a Convolutional Neural Network (CNN) architecture on the CIFAR-10 dataset.

## 🛠️ Tech Stack & Concepts Demonstrated
* **Framework:** PyTorch (`torch`, `torchvision`)
* **Architecture:** Custom CNN with `Conv2d` layers and `MaxPool2d` spatial reduction
* **Optimization:** Adam Optimizer
* **Loss Function:** Cross-Entropy Loss (Multi-class classification)
* **Execution:** CUDA-accelerated GPU training loop

## 📊 Pipeline Pipeline Overview
1. **Data Ingestion:** Downloading and normalising CIFAR-10 image tensors via `torchvision.transforms`.
2. **Model Design:** Building an imperative feature-extraction structure with fully-connected classification linear layers.
3. **Training & Backpropagation:** Tracking loss minimisation over custom epochs with dynamic gradient clearing (`optimizer.zero_grad()`).
4. **Evaluation:** Assessing prediction accuracy across 10,000 hidden test samples.
5. **Serialization:** Exporting optimized parameters as a `.pth` state dictionary file.
