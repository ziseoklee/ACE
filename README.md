# On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering

**Authors:** Ziseok Lee, Minyeong Hwang, Sanghyun Jo, Wooyeol Lee, Jihyung Ko, Young Bin Park, Jae-Mun Choi, Eunho Yang, Kyungsu Kim

[![arXiv](https://img.shields.io/badge/arXiv-2512.10339-b31b1b.svg)](https://arxiv.org/abs/2512.10339)
[![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/pdf/2512.10339)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://ziseoklee.github.io/projects/ACE/)
[![GitHub Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/ziseoklee/ACE/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziseoklee/ACE/blob/main/ace_demo.ipynb)
[![GitHub Stars](https://img.shields.io/github/stars/ziseoklee/ACE?style=social)](https://github.com/ziseoklee/ACE/)

## Abstract

Inference-time steering enables pretrained diffusion/flow models to be adapted to new tasks without retraining. A widely used approach is the ratio-of-densities method, which defines a time-indexed target path by reweighting probability-density trajectories from multiple models. This construction, however, harbors a critical failure mode: **Marginal Path Collapse**, where intermediate densities become non-normalizable even though endpoints remain valid.

This collapse arises systematically when composing heterogeneous models trained on different noise schedules or datasets. In this work, we provide a novel and complete solution:
1.  **Path Existence Criterion**: We derive a criterion that predicts exactly when collapse occurs based on noise schedules and exponents.
2.  **Adaptive path Correction with Exponents (ACE)**: We introduce ACE, which extends Feynman-Kac steering to time-varying exponents to guarantee a valid probability path.

## [Need to Update] Repository Structure

This repository is organized into three main sections corresponding to the experiments in the paper:

* **`toy_experiments/`**: Source code for the Synthetic 2D benchmark demonstrating the path existence criterion and collapse modes.
* **`molecule_experiments/`**: Implementation of ACE for flexible-pose scaffold decoration and molecular design tasks using heterogeneous models.
* **`image_experiments/`**: Experiments applying diffusion steering and ACE to image generation tasks.

*Please refer to the `README.md` within each subdirectory for specific usage instructions and pretrained model checkpoints.*

## Installation

We provide a shared environment for all experiments.

**Prerequisites**
For the toy experiments, run a pretraining code which will save pretrained checkpoints under the directory `/PretrainedToyModels`. Training takes around 30 minutes on a single A6000 GPU.
```bash
python ace_lib/train_toy_models.py
```

**Setup**

```bash
# Clone the repository
git clone https://github.com/ziseoklee/ACE.git
cd ACE

# Create a virtual environment
python3 -m venv ace_env
source ace_env/bin/activate  # On Windows use `ace_env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Toy Experiments
To reproduce the 2D synthetic benchmark results (Figure [TODO]):

```bash
cd 2d_synthetic
# [TODO]
```

### 2. Molecule Experiments
To run the flexible-pose scaffold decoration:
```bash
cd scaffold_decoration
# [TODO]
```

### 3. Image Experiments
To run the image steering corrections:

```bash
cd image_gen
# [TODO]
```

## Citation
If you find this code or our paper useful for your research, please cite:

```
@article{lee2025collapse,
  title={On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering},
  author={Lee, Ziseok and Hwang, Minyeong and Jo, Sanghyun and Lee, Wooyeol and Ko, Jihyung and Park, Young Bin and Choi, Jae-Mun and Yang, Eunho and Kim, Kyungsu},
  journal={arXiv preprint arXiv:2512.10339},
  year={2025}
}
```