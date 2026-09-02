import unittest

import torch
from torch import nn, optim

from optims.config import normalize_cflat_args
from optims.c_flat import C_Flat


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, inputs):
        return self.linear(inputs)


class CFlatModeTest(unittest.TestCase):
    def build_optimizer(self, strategy, **kwargs):
        model = TinyModel()
        base_optimizer = optim.SGD(model.parameters(), lr=0.1)
        optimizer = C_Flat(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            strategy=strategy,
            **kwargs,
        )
        calls = {"count": 0}

        def loss_fn():
            calls["count"] += 1
            logits = model(torch.ones(2, 2))
            loss = torch.square(logits - 1).mean()
            return logits, [loss]

        optimizer.set_closure(loss_fn)
        return optimizer, calls

    def test_mode_normalization(self):
        args = {
            "cflat_mode": "cflat-plus",
            "cflat_A": None,
            "cflat_k": None,
            "cflat_t0": None,
            "cflat_cof": None,
        }
        normalize_cflat_args(args)
        self.assertTrue(args["cflat"])
        self.assertEqual(args["cflat_A"], 5.0)
        self.assertEqual(args["cflat_k"], 0.01)
        self.assertEqual(args["cflat_t0"], 80)
        self.assertEqual(args["cflat_cof"], 0.005)

    def test_off_uses_one_gradient_evaluation(self):
        optimizer, calls = self.build_optimizer("off")
        optimizer.step()
        self.assertEqual(calls["count"], 1)

    def test_basic_uses_full_cflat_step(self):
        optimizer, calls = self.build_optimizer("basic")
        optimizer.step()
        self.assertEqual(calls["count"], 4)

    def test_plus_can_skip_full_cflat_step(self):
        optimizer, calls = self.build_optimizer("plus", A=100.0, k=0.1, t0=-1000)
        optimizer.step()
        self.assertEqual(calls["count"], 1)

    def test_plus_can_run_full_cflat_step(self):
        optimizer, calls = self.build_optimizer("plus", A=0.0)
        optimizer.step()
        self.assertEqual(calls["count"], 4)


if __name__ == "__main__":
    unittest.main()
