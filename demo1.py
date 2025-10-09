def compute_model_size(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    **kwargs
):
    total_params = 0
    total_params += vocab_size * d_model  # token embedding
    total_params += num_layers * ( # per block
        4 * (d_model * d_model) +  # Q, K, V, O projections
        3 * (d_model * d_ff) +     # SwiGLU layers
        2 * (d_model)              # RMSNorm layers
    )
    total_params += d_model  # final RMSNorm
    total_params += d_model * vocab_size  # output projection
    return total_params

def compute_memory(vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    **kwargs
) -> float:
    param_size = compute_model_size(vocab_size, context_length, d_model, num_layers, num_heads, d_ff)
    optimizer_size = param_size * 3  # Adam optimizer
    activation_size = batch_size * context_length *(
        num_layers * (8 * d_model + 2 * d_ff + 2 * num_heads * context_length) + d_model + 2 * vocab_size  
    )
    memory_bytes = (param_size + optimizer_size + activation_size) * 4  # assuming 4 bytes per parameter (float32)
    return memory_bytes / (1024 ** 3)  # convert to GB


if __name__ == "__main__":
    # GPT-2 XL
    cfg = dict(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400, 
        rope_theta=10000,
        batch_size=0,
    )
    # My
    cfg = dict(
        vocab_size=32000,
        context_length=256,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048, 
        rope_theta=10000,
        batch_size=128,
    )
    model_size = compute_model_size(**cfg)
    memory_gb = compute_memory(**cfg)
    print(f"Model size: {model_size/(1024**3):.2f}B parameters")
    print(f"Estimated memory usage during training: {memory_gb:.2f} GB")
