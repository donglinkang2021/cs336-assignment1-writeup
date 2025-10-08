def compute_gpt2_params(d_model:int, n_layer:int) -> float:
    vocab_size = 50_257
    d_ff = 4 * d_model
    # return vocab_size * d_model + n_layer * (4 * d_model * d_model + 2 * d_model * d_ff) + d_model * vocab_size # original gpt-2 formula
    return vocab_size * d_model + n_layer * (4 * d_model * d_model + 3 * d_model * d_ff) + d_model * vocab_size # our use swiglu gpt-2 formula


print("GPT-2 small", compute_gpt2_params(768, 12) / 1024 / 1024, "M params")
print("GPT-2 medium", compute_gpt2_params(1024, 24) / 1024 / 1024, "M params")
print("GPT-2 large", compute_gpt2_params(1280, 36) / 1024 / 1024, "M params")
print("GPT-2 XL", compute_gpt2_params(1600, 48) / 1024 / 1024 / 1024, "B params")

def compute_gpt2_flops(d_model:int, n_layer:int, T:int=1024) -> dict:
    d_ff = 4 * d_model
    vocab_size = 50_257
    attn_flops = 4 * 2 * T * d_model * d_model + 2 * 2 * T * T * d_model
    ff_flops = 3 * 2 * T * d_model * d_ff
    lm_head_flops = 2 * T * d_model * vocab_size
    total_flops = n_layer * (attn_flops + ff_flops) + lm_head_flops
    return {
        "attention": n_layer * attn_flops / total_flops,
        "feed-forward": n_layer * ff_flops / total_flops,
        "lm-head": lm_head_flops / total_flops,
        "total": total_flops / 1e12, # in TFLOPs
    }, {
        "attention": n_layer * attn_flops / 1e12,
        "feed-forward": n_layer * ff_flops / 1e12,
        "lm-head": lm_head_flops / 1e12,
        "total": total_flops / 1e12,
    }

print("GPT-2 small", compute_gpt2_flops(768, 12))
print("GPT-2 medium", compute_gpt2_flops(1024, 24))
print("GPT-2 large", compute_gpt2_flops(1280, 36))
print("GPT-2 XL", compute_gpt2_flops(1600, 48))
print("GPT-2 XL", compute_gpt2_flops(1600, 48, 16384))

"""output:
GPT-2 small 181.61865234375 M params
GPT-2 medium 482.158203125 M params
GPT-2 large 1022.69775390625 M params
GPT-2 XL 1.9808322191238403 B params
GPT-2 small ({'attention': 0.276396942718713, 'feed-forward': 0.49751449689368343, 'lm-head': 0.22608856038760353, 'total': 0.349630365696}, {'attention': 0.09663676416, 'feed-forward': 0.173946175488, 'lm-head': 0.079047426048, 'total': 0.349630365696})
GPT-2 medium ({'attention': 0.29932707434661254, 'feed-forward': 0.5986541486932251, 'lm-head': 0.1020187769601624, 'total': 1.033109504}, {'attention': 0.309237645312, 'feed-forward': 0.618475290624, 'lm-head': 0.105396568064, 'total': 1.033109504})
GPT-2 large ({'attention': 0.2996151010432329, 'feed-forward': 0.6420323593783562, 'lm-head': 0.05835253957841083, 'total': 2.2577545216}, {'attention': 0.67645734912, 'feed-forward': 1.4495514624, 'lm-head': 0.13174571008, 'total': 2.2577545216})
GPT-2 XL ({'attention': 0.29440647731422626, 'feed-forward': 0.6691056302596051, 'lm-head': 0.036487892426168594, 'total': 4.5133365248}, {'attention': 1.3287555072, 'feed-forward': 3.01989888, 'lm-head': 0.1646821376, 'total': 4.5133365248})    
GPT-2 XL ({'attention': 0.65922723665908, 'feed-forward': 0.32315060620543135, 'lm-head': 0.017622157135488675, 'total': 149.5227957248}, {'attention': 98.5694994432, 'feed-forward': 48.31838208, 'lm-head': 2.6349142016, 'total': 149.5227957248})
"""