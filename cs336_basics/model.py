import torch
from torch import nn
from einops import einsum

def get_std(
    in_feature: int,
    out_feature: int
):
    return torch.sqrt(torch.tensor(2)/(in_feature + out_feature))

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device=None,
            dtype: torch.device=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        std = get_std(in_features, out_features)
        self.W = nn.init.trunc_normal_(self.W, mean = 0, std=std, a=-3*std, b=3*std)

    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return einsum(x, self.W,  "... in_feature, out_feature in_feature -> ... out_feature")

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device=None,
        dtype: torch.device=None
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        std = get_std(num_embeddings, embedding_dim)
        self.weights = nn.init.trunc_normal_(self.weights, mean = 0, std=std, a=-3*std, b=3*std)
    
    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        return self.weights[token_ids]

