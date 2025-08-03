import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional


def lr_cosine_schedule(
    t: int,
    a_max: float,
    a_min: float,
    warmup_inter: int,
    cs_iter: int
) -> float:
    # warmup
    if t <= warmup_inter:
        return t/warmup_inter*a_max
    elif t <= cs_iter:
        return a_min + 0.5*(1 + math.cos((t - warmup_inter)/(cs_iter - warmup_inter)*math.pi))*(a_max - a_min)
    else:
        return a_min


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group["lr"]
            betas_1, betas_2 = group["betas"]
            weight_deacy =  group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                m = betas_1 * m + ((1 - betas_1)*g)
                v = betas_2 * v + ((1 - betas_2)*(torch.square(g)))

                lr_t = lr * (math.sqrt(1 - (betas_2**t)) / (1 - (betas_1**t)))
                p.data -= lr_t *  m / (torch.sqrt(v) + eps)

                p.data -= lr * weight_deacy * p.data

                import pdb; pdb.set_trace()

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
