# C-Flat++

This repository contains the official implementation of "C-Flat++: Towards a More Efficient and Powerful Framework for Continual Learning" [[paper](https://arxiv.org/abs/2508.18860)], accepted by IJCV.

## 🥳 Integration Notice 🥳

*C-Flat* has been fully integrated into the [[INFTY](https://github.com/THUDM/INFTY)] package, serving as a strong algorithm supporting generalizability within the continual learning ecosystem. For the latest version and extended functionalities, please refer to **INFTY**.

## Acknowledgment
This repository is partially based on [PyCIL](https://github.com/G-U-N/PyCIL).

## Prerequisites
This code is implemented in PyTorch, and we have tested the code under the following environment settings:
- python = 3.9.13
- torch = 2.0.1
- torchvision = 0.15.2

Create the project environment from the tracked environment file:
```
conda env create -f environment.yml
conda activate cflat-plus
```

## Usage
Step 0:  Datasets

We provide the source code on three benchmark datasets: CIFAR-100, ImageNet-100, and Tiny-ImageNet. For CIFAR-100, it will be downloaded automatically upon first use. For the other two datasets or self-made datasets, please modify the corresponding path in `utils/data.py`.

Step 1: Quick start

Train CIFAR-100 on [WA](https://arxiv.org/abs/1911.07053) with the baseline optimizer:
```
python main.py --config=exps/wa.json
```

Switch to C-Flat with the original command:
```
python main.py --config=exps/wa.json --cflat
```

Switch to C-Flat++ without changing the experiment config:
```
python main.py --config=exps/wa.json --cflat-plus
```

The equivalent unified interface is `--cflat-mode {off,cflat,cflat-plus}`. C-Flat++ exposes the controller parameters through `--cflat-a`, `--cflat-k`, `--cflat-t0`, and `--cflat-cof`; their paper defaults are `5.0`, `0.01`, `80`, and `0.005`, respectively. The shared C-Flat parameters default to `rho=0.2` and `lambda=0.2`.

For other detailed model configurations, please refer to `exps/[model_name].json`.

Step 2:  Custom application

2.1 Create a closure to calculate the loss
```
def create_loss_fn(self, inputs, targets):
    def loss_fn():
        logits = self._network(inputs)["logits"]
        loss_clf = F.cross_entropy(logits, targets)
        return logits, [loss_clf]
    return loss_fn
```

2.2 Introduce the cflat optimizer
```
optimizer = C_Flat(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, cflat=True)
```

For C-Flat++, select the `plus` strategy:
```
optimizer = C_Flat(
    params=self._network.parameters(),
    base_optimizer=base_optimizer,
    model=self._network,
    strategy="plus",
)
```

2.3 Model update
```
loss_fn = self.create_loss_fn(inputs, targets)
optimizer.set_closure(loss_fn)
logits, loss_list = optimizer.step()
```

## Citation

If you find this repo useful for your research, please consider citing the paper.

```
@article{bian2024make,
  title={Make continual learning stronger via c-flat},
  author={Bian, Ang and Li, Wei and Yuan, Hangjie and Wang, Mang and Zhao, Zixiang and Lu, Aojun and Ji, Pengliang and Feng, Tao and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={7608--7630},
  year={2024}
}
```
## Contact

If there are any questions, please feel free to contact the corresponding author (ymjiii98@gmail.com).
