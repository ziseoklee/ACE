# [ICML 2026] On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering

**Authors:** Ziseok Lee, Minyeong Hwang, Wooyeol Lee, Sanghyun Jo, Jihyung Ko, Young Bin Park, Jae-Mun Choi, Eunho Yang, Kyungsu Kim

[![arXiv](https://img.shields.io/badge/arXiv-2512.10339-b31b1b.svg)](https://arxiv.org/abs/2512.10339)
[![PDF](https://img.shields.io/badge/PDF-Download-red)](https://arxiv.org/pdf/2512.10339)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://ziseoklee.github.io/projects/ACE/)
[![GitHub Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/ziseoklee/ACE/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziseoklee/ACE/blob/main/ace_demo.ipynb)
[![GitHub Stars](https://shields.io/github/stars/ziseoklee/ACE?style=social)](https://github.com/ziseoklee/ACE/)


## Abstract

Inference-time steering enables pretrained diffusion/flow models to be adapted to new tasks without retraining. A widely used approach is the ratio-of-densities method, which defines a time-indexed target path by reweighting probability-density trajectories from multiple models. This construction, however, harbors a critical failure mode: **Marginal Path Collapse**, where intermediate densities become non-normalizable even though endpoints remain valid.

This collapse arises systematically when composing heterogeneous models trained on different noise schedules or datasets. In this work, we provide a novel and complete solution:
1.  **Path Existence Criterion**: We derive a criterion that predicts exactly when collapse occurs based on noise schedules and exponents.
2.  **Adaptive path Correction with Exponents (ACE)**: We introduce ACE, which extends Feynman-Kac steering to time-varying exponents to guarantee a valid probability path.

## Repository Structure

This repository is organized into three main sections corresponding to the experiments in our paper:

* **`2d_synthetic/`**: Scripts for the Synthetic 2D benchmark demonstrating the path existence criterion, path collapse, and how ACE repairs collapsed paths.
* **`molecule/`**: Implementation of ACE for drug design tasks presented in the paper: scaffold decoration and fragment linking.
* **`image/`**: Experiments applying ACE to compositional image generation. We are finalizing the code for the image experiments, which will be available by June 5th.

*Detailed `README.md` files and instructions will be provided within each subdirectory.*

## Installation

We provide a shared environment for all experiments. All subdirectories assume that `ace_env` has been activated.

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

## Citation
If you find this code or our paper useful for your research, please cite:

```bibtex
@inproceedings{
lee2026collapse,
title={On the Collapse of Generative Paths: A Criterion and Correction for Diffusion Steering},
author={Ziseok Lee and Minyeong Hwang and Wooyeol Lee and Jihyung Ko and Young Bin Park and Jae-Mun Choi and Eunho Yang and Kyungsu Kim},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=emv2qsi3TG}
}
```
