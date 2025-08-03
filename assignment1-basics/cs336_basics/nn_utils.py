from jaxtyping import Float, Int
import torch
from torch import Tensor
from einops import rearrange, reduce
from typing import Iterable



def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    rescaled_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp_f = torch.exp(rescaled_x)
    return exp_f / torch.sum(exp_f, dim=dim, keepdim=True)


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    # for numerica stability calculate max for each bath_size
    input_max = reduce(inputs, 'b v -> b 1', 'max')
    input_reduce = inputs - input_max
    log_sum_exp = torch.log(reduce(torch.exp(input_reduce), "b v -> b", 'sum'))
    targets = rearrange(targets, "b -> b 1")
    batch_indices = rearrange(torch.arange(input_reduce.shape[0]), "b -> b 1")
    targets_prob  = rearrange(input_reduce[batch_indices, targets], "b 1 -> b")
    ce_loss = - targets_prob + log_sum_exp
    import pdb; pdb.set_trace()
    return reduce(ce_loss, "b -> ", 'mean')


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    global_param_sum = 0.0

    for grad in grads:
        global_param_sum += (grad**2).sum()

    norm = torch.sqrt(global_param_sum)
    clip_coef = min(1, max_l2_norm / (norm + eps))
    for g in grads:
        g *= clip_coef
