"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import datetime
from pathlib import Path
import torch

from aurora.batch import Batch, Metadata
from aurora.model.aurora_lite import AuroraLite
from aurora.model.decoder_lite import MLPDecoderLite

from data_prep import getBatch

def loss(pred: Batch) -> torch.Tensor:
    """A sample loss function. You should replace this with your own loss function."""
    surf_values = pred.surf_vars.values()
    atmos_values = pred.atmos_vars.values()
    return sum((x * x).sum() for x in tuple(surf_values) + tuple(atmos_values))

batch = getBatch(Path("./data/downloads"))

surf_vars_new = ["tp_mswep", # total precipitation MSWEP
                "pe", # potential evaporation
                "r", # runoff
                "swc", # soil water content,
                ]

modelAurora = AuroraLite(
    use_lora=False, 
    autocast=True, #Use AMP (mixed precision to fit to gpu)
    surf_vars=("2t", "10u", "10v", "msl"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"))

modelAurora.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
modelAurora = modelAurora.to("cuda")
modelAurora.eval()

modelDecoder = MLPDecoderLite(surf_vars_new=surf_vars_new, 
                                        patch_size=modelAurora.decoder.patch_size, 
                                        embed_dim=2*modelAurora.encoder.embed_dim,
                                        hidden_dims=[512, 512, 256],
                                        )
checkpoint = torch.load("./lite-decoder.ckpt")
modelDecoder.load_state_dict(checkpoint)
modelDecoder.train()
modelDecoder.to("cuda")
# modelDecoder.eval()

opt = torch.optim.AdamW(modelDecoder.parameters(), lr=3e-4)

for i in range(10):
    print(f"Step {i}")

    with torch.inference_mode():
        preds_org, lat_dec = modelAurora.forward(batch)
        latent_decoder = lat_dec.detach().clone()

    opt.zero_grad()
    preds_new = modelDecoder.forward(latent_decoder, batch.metadata.lat, batch.metadata.lon)
    loss_value = loss(preds_new)
    loss_value.backward()
    opt.step()
