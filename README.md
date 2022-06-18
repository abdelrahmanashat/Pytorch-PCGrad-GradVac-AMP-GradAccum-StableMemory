# Pytorch-PCGrad-AMP-GradAccum

PyTorch 1.11 reimplementation of [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) with Automatic Mixed Precision Training and Gradient Accumulation

## Setup
Install the required packages via:
```
pip install -r requirements.txt
```

## Usage

```python
import torch
from pcgrad_amp import PCGradAMP

ACCUM_STEPS = ...
NUM_EPOCHS = ...
model = ...
train_loader = ...

optimizer = torch.optim.Adam(model.parameters())
scaler = torch.cuda.amp.GradScaler()
num_tasks = 2
optimizer = PCGradAMP(num_tasks, optimizer, scaler=scaler, reduction='sum', cpu_offload= True)

total_steps = 0
for ep in range(NUM_EPOCHS):
    for batch in train_loader:
        losses = [...]
        optimizer.backward(losses) # compute & accumulate gradients
        total_steps += 1
        if total_steps % ACCUM_STEPS:
            optimizer.step() # parameter update

```

## Training
- Multi-MNIST
  Please run the training script via the following command. Part of implementation is leveraged from https://github.com/intel-isl/MultiObjectiveOptimization
  ```
  python main_multi_mnist_amp.py
  ```
  The result is shown below.
  | Method                  | left-digit | right-digit |
  | ----------------------- | ---------: | ----------: |
  | Jointly Training        |      90.30 |       90.01 |
  | Pytorch-PCGrad (Wei-Cheng Tseng) |  95.00 |   92.00 |
  | **Pytorch-PCGrad-AMP-GradAccum (this repo.)** |  **95.00** |   **92.00** |
  | PCGrad (original paper)       |      96.58 |       95.50 |

## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={arXiv preprint arXiv:2001.06782},
  year={2020}
}

@misc{Pytorch-PCGrad,
  author = {Wei-Cheng Tseng},
  title = {WeiChengTseng/Pytorch-PCGrad},
  url = {https://github.com/WeiChengTseng/Pytorch-PCGrad.git},
  year = {2020}
}

@misc{Pytorch-PCGrad-AMP-GradAccum,
  author = {Antoine Nzeyimana},
  title = {Pytorch-PCGrad-AMP-GradAccum/Antoine Nzeyimana},
  url = {https://github.com/anzeyimana/Pytorch-PCGrad-AMP-GradAccum},
  year = {2022}
}
```