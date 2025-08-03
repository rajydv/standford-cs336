# # Using submitit for parameter sweep
# import submitit

# def train_model(model_size, context_length):
#     # Your training code here
#     pass

# # Setup parameter grid
# sizes = [128, 256, 512]
# lengths = [512, 1024, 2048]

# # Create executor
# executor = submitit.AutoExecutor(folder="log_dir")
# executor.update_parameters(
#     slurm_partition="gpu",
#     gpus_per_node=1
# )

# # Submit jobs for all combinations
# jobs = []
# for size in sizes:
#     for length in lengths:
#         job = executor.submit(train_model, size, length)
#         jobs.append(job)

# # Get results
# results = [job.result() for job in jobs]


# Need a function to intialize a model with given configuration
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit
from torch import nn
import typer
from typing import Annotated
import pandas as pd
import click
from tqdm.auto import tqdm
from einops import einsum, rearrange

def get_model_run_time(
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: str,
    backward_pass: False
) -> dict[str, int | float]:
    model = BasicsTransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta
    )

    X = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
    Y = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    s_time = timeit.default_timer()

    # forward pass
    output = model(X)

    forward_time = timeit.default_timer() - s_time

    if backward_pass:
        output_flat = rearrange(output, "b sql v -> (b sql) v")
        target_flat = rearrange(Y, "b sql -> (b sql)")
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output_flat, target_flat)
        # compute gradient / backward pass
        loss.backward()
    
    e_time = timeit.default_timer()

    elapsed_time = e_time - s_time
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "backward_pass_included": backward_pass,
        "device": device,
        "forward_time": forward_time,
        "backward_time": e_time - forward_time,
        "total_time": elapsed_time
    }
    

def run_benchmark(
    warmup_steps: Annotated[int, typer.Argument(help="Number of warmup steps")],
    steps: Annotated[int, typer.Argument(help="Number of measurement steps")],
    backward_pass: Annotated[bool, typer.Option(help="Include backward pass")] = False
) -> str:
    data  = []
    d_models = [16, 32]
    d_ffs = [64, 128]
    num_layers = [4]
    num_heads = [4]
    context_length = 64
    batch_size = 1024*1
    vocab_size = 1000
    rope_theta = 10000

    device  = "cpu"

    for d_model in tqdm(d_models, desc="Embedding Dim Size"):
        for d_ff in tqdm(d_ffs, desc="FF Size", leave=False):
            for num_layer in tqdm(num_layers, desc="Layers", leave=False):
                for num_head in tqdm(num_heads, desc="Heads", leave=False):
                    for _ in range(1, warmup_steps+1):
                        run_data = get_model_run_time(
                            batch_size,
                            vocab_size,
                            context_length,
                            d_model,
                            num_layer,
                            num_head,
                            d_ff,
                            rope_theta,
                            device,
                            backward_pass
                        )
                        run_data.update({
                            "is_warmup": True
                        })
                        data.append(run_data)

                    for _ in range(1, steps+1):
                        run_data = get_model_run_time(
                            batch_size,
                            vocab_size,
                            context_length,
                            d_model,
                            num_layer,
                            num_head,
                            d_ff,
                            rope_theta,
                            device,
                            backward_pass
                        )
                        run_data.update({
                            "is_warmup": False
                        })
                        data.append(run_data)

    df = pd.DataFrame(data)
    click.secho(f"Writing len: {len(data)} data in file ")
    suffix = f"{warmup_steps}_{steps}_{backward_pass}"
    with open(f"benchmark_output_{suffix}.md", "w") as f:
        f.write(df.to_markdown())

if __name__ == "__main__":
    typer.run(run_benchmark)
