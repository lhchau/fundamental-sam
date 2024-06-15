## Fundamental Convergence Analysis of Sharpness-Aware Minimization

**Pham Duy Khanh, Hoang-Chau Luong, Boris S. Mordukhovich, Dat Ba Tran**

This repo implements paper ["Fundamental Convergence Analysis of Sharpness-Aware Minimization"](https://arxiv.org/pdf/2401.08060.pdf) in Pytorch. SAM's implementation based on [davda54/sam](https://github.com/davda54/sam).

### Abstract

The paper investigates the fundamental convergence properties of Sharpness-Aware Minimization (SAM), a recently proposed gradient-based optimization method (Foret et al., 2021) that significantly improves the generalization of deep neural networks. The convergence properties including the stationarity of accumulation points, the convergence of the sequence of gradients to the origin, the sequence of function values to the optimal value, and the sequence of iterates to the optimal solution are established for the method. The universality of the provided convergence analysis based on inexact gradient descent frameworks (Khanh et al., 2023b) allows its extensions to the normalized versions of SAM such as VaSSO (Li & Giannakis, 2023), RSAM (Liu et al., 2022), and to the unnormalized versions of SAM such as USAM (Andriushchenko & Flammarion, 2022). Numerical experiments are conducted on classification tasks using deep learning models to confirm the practical aspects of our analysis.

### Getting Started

#### Installation

You need to have Anaconda installed on your machine. To install the dependencies, you can use a package manager like `conda` and `pip`:

```bash
conda env create -f environment.yml
conda activate fundamental-sam
pip install -e .
```

You need to install WandB for experiment tracking:

```bash
pip install wandb
wandb login
```

#### Running the Script

Choose any file name in `./config` and replace it in the below bash.

```bash
python fundamental_sam/train_sam.py --experiment=config/file/name
```
