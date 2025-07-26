# Deep Learning CNN Demo Notebooks

This repository contains PyTorch-based Jupyter Notebooks demonstrating and comparing three convolutional neural network (CNN) architectures on popular image classification datasets:

- **Custom CNN Architecture (My_Demo_Architecture)**
- **ResNet-18 (02_ResNet_Demo)**
- **LeNet-5 (02_LeNet_Demo)**

Each notebook is fully self-contained and guides you through data loading, model definition, training, evaluation, and visualization.

## Table of Contents

- [Requirements](#requirements)
- [Notebook Overviews](#notebook-overviews)
  - [My_Demo_Architecture.ipynb](#my_demo_architectureipynb)
  - [02_ResNet_Demo.ipynb](#02_resnet_demoipynb)
  - [02_LeNet_Demo.ipynb](#02_lenet_demoipynb)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Requirements

- Python 3.7+
- Jupyter Notebook / JupyterLab
- PyTorch (>= 1.7.0)
- torchvision
- matplotlib
- [Optional, for Colab] Google Colab environment

Install dependencies (recommended):
```bash
pip install torch torchvision matplotlib notebook
```

## Notebook Overviews

### My_Demo_Architecture.ipynb

- **Dataset:** CIFAR-10 (Images resized to 32x32 with augmentations)
- **Architecture:** A custom CNN featuring multi-branch convolutional "Intermediate Blocks" with dynamic channel weighting, batch normalization, and dropout, followed by a global pooling output block.
- **Workflow:**
  - Data loading and augmentation
  - Model construction with modular blocks
  - Xavier initialization
  - Training with Adam optimizer and a learning rate scheduler
  - Evaluation loop and metrics plotting
- **Goal:** Explore creative CNN architectures and modular construction for image classification.

### 02_ResNet_Demo.ipynb

- **Dataset:** Fashion-MNIST (resized to 224x224, grayscale)
- **Architecture:** Reproduction of ResNet-18 using manually implemented residual blocks and macroblocks with skip connections, batch normalization, and adaptive pooling.
- **Workflow:**
  - Data loading and visualization
  - Step-wise implementation of residual blocks emulating the original ResNet-18
  - Training accuracy assessment (supports GPU/Colab execution)
  - Optional end-to-end training loop provided
- **Goal:** Learn and demystify the ResNet-18 architecture by implementing it from scratch.

### 02_LeNet_Demo.ipynb

- **Dataset:** Fashion-MNIST (28x28 grayscale)
- **Architecture:** Classic LeNet-5 model using sigmoid activations, average pooling, and three fully connected layers.
- **Workflow:**
  - Data loading and batch visualization
  - Sequential model definition using PyTorch modules
  - Training loop, validation, and plotting of accuracy and loss per epoch
- **Goal:** Illustrate fundamental CNN design and training, and serve as a baseline for comparison.

## How to Run the Notebooks

1. **Clone this repository:**
   ```bash
   git clone 
   cd 
   ```

2. **Install dependencies** using pip (see [Requirements](#requirements)).

3. **Start Jupyter Notebook or upload notebook(s) to Colab:**
   ```bash
   jupyter notebook
   ```
   - Alternatively, open in Google Colab for free GPU acceleration.

4. **Open a notebook file and run the cells sequentially.**  
   Each notebook is self-contained and will download its required datasets automatically.

## Project Structure

```
.
├── My_Demo_Architecture.ipynb    # Custom CNN model, CIFAR-10 (32x32 RGB)
├── 02_ResNet_Demo.ipynb          # ResNet-18 implementation, Fashion-MNIST (224x224 grayscale)
├── 02_LeNet_Demo.ipynb           # LeNet-5 implementation, Fashion-MNIST (28x28 grayscale)
└── README.md                     # This file
```

## Citation

Please cite the original PyTorch and torchvision teams if you use these resources in your research or coursework.

## License

Distributed under the MIT License (see [LICENSE](LICENSE) file if provided; otherwise, default to MIT for academic demo purposes).

## Acknowledgments

- PyTorch and torchvision open-source projects
- Original publications:
  - **LeNet-5**: [Yann LeCun et al., 1998](http://yann.lecun.com/exdb/lenet/)
  - **ResNet**: [Kaiming He et al., 2015](https://arxiv.org/abs/1512.03385)
- Fashion-MNIST and CIFAR-10 datasets

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/26366238/11c1c6eb-2b65-45e3-909c-e0b689f6eaac/My_Demo_Architecture.ipynb
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/26366238/3644e8a7-ac0c-4ab9-bfe8-55e95e91a4de/02_ResNet_Demo.ipynb
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/26366238/4315af01-ab41-4f23-aaac-11f9a78bdea5/02_LeNet_Demo.ipynb
