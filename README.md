# Pytorch-PCGrad-GradVac-AMP-GradAccum-StableMemory

PyTorch 1.11 reimplementation of multi task gradient adaptation ideas with no memory leaks from papers:
- [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)
- [Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models](https://arxiv.org/abs/2010.05874)

Supports:
- Automatic mixed precision (AMP)
- Gradient Accumulation (with CPU offload support)

Fixes:
- Out of Memory (OOM) problems associated with the original implementation written by respectable authors Wei-Cheng Tseng and Antoine Nzeyimana. There were undeleted tensors that quickly fill up the GPU memory leading to OOM.

Adaptation from the following repositories:
- https://github.com/WeiChengTseng/Pytorch-PCGrad
- https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum
- https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/GradVac.py

## Setup
Install the required packages via:
```
pip install -r requirements.txt
```

## Usage

```python
import torch
from pcgrad_amp import PCGradAMP
from gradvac_amp import GradVacAMP

DEVICE = ...
ACCUM_STEPS = ...
NUM_EPOCHS = ...
LR = ...
BATCH_SIZE = ...

model = ...
train_loader = ...

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# lr_scheduler = ...
scaler = torch.cuda.amp.GradScaler()
num_tasks = 2

# For Gradient Surgery/PCGrad
#grad_optimizer = PCGradAMP(num_tasks, optimizer, scaler=scaler, reduction='sum', cpu_offload= False)

# For Gradient Vaccine
grad_optimizer = GradVacAMP(num_tasks, optimizer, DEVICE, scaler = scaler, beta = 1e-2, reduction='sum', cpu_offload = False)

total_steps = 0
for ep in range(NUM_EPOCHS):
    for mini_batch in train_loader:
        losses = [...]
        grad_optimizer.backward(losses) # Compute & Accumulate gradients
        total_steps += 1
        if (total_steps % ACCUM_STEPS) == 0:
            # lr_scheduler.step()
            grad_optimizer.step() # Parameter update step

```


## Reference

Please cite as:

```
@article{yu2020gradient,
  title={Gradient surgery for multi-task learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={5824--5836},
  year={2020}
}

@article{wang2020gradient,
  title={Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models},
  author={Wang, Zirui and Tsvetkov, Yulia and Firat, Orhan and Cao, Yuan},
  journal={arXiv preprint arXiv:2010.05874},
  year={2020}
}

@misc{Pytorch-PCGrad-GradVac-AMP-GradAccum-StableMemory,
  author = {Abdelrahman Nashat},
  title = {Pytorch-PCGrad-GradVac-AMP-GradAccum-StableMemory/Abdelrahman Nashat},
  url = {https://github.com/anzeyimana/Pytorch-PCGrad-GradVac-AMP-GradAccum-StableMemory},
  year = {2025}
}
```
