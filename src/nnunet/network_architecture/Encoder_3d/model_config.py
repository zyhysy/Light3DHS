def get_config(mode: str = "xxs") -> dict:
    if mode == "xx_small":
        mv2_exp_mult = 2
        config = {
            "layer4": {  # 14x14
                "out_channels": 64,
                "transformer_channels": 80,
                "ffn_dim": 160,
                "transformer_blocks": 1,
                "patch_h": 2,  # 4,
                "patch_w": 2,  # 4,
                "patch_d": 2,
                "stride": 1,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "layer5": {  # 7x7
                "out_channels": 128,
                "transformer_channels": 144,
                "ffn_dim": 288,
                "transformer_blocks": 1,
                "patch_h": 2,
                "patch_w": 2,
                "patch_d": 2,
                "stride": 1,
                "mv_expand_ratio": mv2_exp_mult,
                "num_heads": 4,
                "block_type": "mobilevit",
            },
            "last_layer_exp_factor": 4,
            "cls_dropout": 0.1
        }
    else:
        raise NotImplementedError

    for k in ["layer4", "layer5"]:
        config[k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})

    return config