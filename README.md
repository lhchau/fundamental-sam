## Fundamental Convergence Analysis of Sharpness-Aware Minimization

**Pham Duy Khanh, Hoang-Chau Luong, Boris S. Mordukhovich, Dat Ba Tran**

This repo implements paper ["Fundamental Convergence Analysis of Sharpness-Aware Minimization"](https://arxiv.org/pdf/2401.08060.pdf) in Pytorch. SAM's implementation based on [davda54/sam](https://github.com/davda54/sam).

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
