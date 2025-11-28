from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import xarray as xr

from aurora import AuroraPretrained, Batch, Metadata
from aurora.normalisation import (
    normalise_surf_var,
    unnormalise_surf_var,
    locations,
    scales,
)
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import SSTDataset, collate_fn

MAE = nn.L1Loss()

# surf_weights = {"MSL": 1.5, "U10": 0.77, "V10": 0.66, "2T": 3.0}

# atmos_weights = {"z": 2.8, "Q": 0.78, "T": 1.7, "U": 0.87, "V": 0.6}


model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "sst", "siconc"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"),
    autocast=True,
)

# normalization (yearly) means
locations["sst"] = 17.1771
locations["siconc"] = 0.1589
locations["sh"] = -12.8700


# normalization (yearly) standard deviations
scales["sst"] = 10.301042
scales["siconc"] = 0.32129946
locations["sh"] = 12.011224


def sensible_heat(
    sst: torch.Tensor,
    t2: torch.Tensor,
    wind_speed: torch.Tensor,
    rhoao: float = 1.2250,  # density at 15C
    cp: float = 1.006e3,  # 1.006kJ
    cs: float = 1e-3,
    # cs: float = 0.9e-3,
):
    return -rhoao * cp * cs * wind_speed * (sst - t2)


def loss(
    pred: Batch,
    target: Batch,
    over_size: int,
    ice_threshold: float = 0.3,
    physics_loss_hyperparameter=0.03,
) -> torch.Tensor:
    """
    Docstring for loss

    :param pred: Batch containing the predicted values
    :type pred: Batch
    :param target: Batch containing the target information. Please note that this also contains all the information required to predict for the next iteration, meaning the data for the last `valid_time` should be used
    :type target: Batch
    :param over_size: 1/(H x W), precomputed for efficiency (very little impact)
    :type over_size: int
    :return: The loss
    :rtype: Tensor
    """
    # TODO check if only last value of target is needed?
    # According to https://microsoft.github.io/aurora/batch.html#model-output, yes
    surf_values = pred.surf_vars.values()
    pred_sst = surf_values["sst"][-1]
    pred_t2 = surf_values["2t"][-1]
    pred_u10 = surf_values["10u"][-1]
    pred_v10 = surf_values["10v"][-1]
    pred_wind_speed = torch.sqrt(pred_u10**2, pred_v10**2)

    # only the last value in target
    target_sst = target["sst"][-1]
    target_t2 = target["2t"][-1]
    target_u10 = target["10u"][-1]
    target_v10 = target["10v"][-1]
    target_wind_speed = torch.sqrt(target_u10**2, target_v10**2)

    pred_sh = sensible_heat(pred_sst, pred_t2, pred_wind_speed)
    target_sh = sensible_heat(target_sst, target_t2, target_wind_speed)

    norm_pred_sst = normalise_surf_var(pred_sst, "sst", model.surf_stats)
    # norm_target_sst = normalise_surf_var(target_sst, "sst", model.surf_stats)
    # phys_loss_pre = abs(norm_pred_sst - norm_target_sst) * abs(pred_sh - target_sh)
    norm_pred_sh = normalise_surf_var(pred_sh, "sh", model.surf_stats)
    norm_target_sh = normalise_surf_var(target_sh, "sh", model.surf_stats)
    phys_loss_pre = abs(norm_pred_sh - norm_target_sh)
    # TODO is this the correct interpretation
    ice_cover = target["siconc"][-1]
    # Removes latent heat for ocean covered by more than `ice_threshold` ice
    phys_loss_pre = torch.where(ice_cover > ice_threshold, torch.nan, phys_loss_pre)

    phys_loss = torch.nansum(phys_loss_pre) * over_size

    mae_loss = nn.functional.mse_loss(norm_pred_sst, target_sst)

    return mae_loss + physics_loss_hyperparameter * phys_loss


# def collate_batches(batch_items) -> tuple[Batch, Batch]:
#     batch
#     for batches in batch_items:
#         surf_vars = {
#             k: torch.cat([b.surf_vars[k] for b in batches], dim=0)
#             for k in batches[0].surf_vars
#         }
#         atmos_vars = {
#             k: torch.cat([b.atmos_vars[k] for b in batches], dim=0)
#             for k in batches[0].atmos_vars
#         }

#         static_vars = batches[0].static_vars  # Same for every sample.

#         metadata = batches[0].metadata
#         metadata = type(metadata)(
#             lat=metadata.lat,
#             lon=metadata.lon,
#             time=tuple(b.metadata.time[0] for b in batches),
#             atmos_levels=metadata.atmos_levels,
#         )

#         stacked_batch = type(batches[0])(
#             surf_vars=surf_vars,
#             static_vars=static_vars,
#             atmos_vars=atmos_vars,
#             metadata=metadata,
#         )


if not torch.cuda.is_available():
    raise RuntimeError("Need CUDA for Aurora fine-tuning.")
device = torch.device("cuda")
data_path = Path("scratch/data/finetune-data-2020-2024")
# data_path = Path("./data/downloads")
dataset = SSTDataset(
    data_path, ["sst"], surface_variables=["2t", "10u", "10v", "msl", "sst", "siconc"]
)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batches)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, collate_fn=collate_fn)

# pretrained_weights = torch.load()

model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)

# TODO figure out token embeds
#github.com/microsoft/aurora/issues/24
model.encoder.surf_token_embeds.weights["sst"] =nn.Parameter(torch.zeros(model.encoder.surf_token_embeds.embed_dim, 1, *model.encoder.surf_token_embeds.kernel_size))
# model.encoder.surf_token_embeds.weights["siconc"] = torch.zeros(model.encoder.surf_token_embeds.weights["siconc"].shape)
model.encoder.surf_token_embeds.weights["siconc"] =nn.Parameter(torch.zeros(model.encoder.surf_token_embeds.embed_dim, 1, *model.encoder.surf_token_embeds.kernel_size))
# model.encoder.surf_token_embeds.bias["sst"] = torch.zeros(model.encoder.surf_token_embeds.bias["sst"].shape)
# model.encoder.surf_token_embeds.bias["sst"] = nn.Parameter(torch.zeros(model.encoder.surf_token_embeds.embed_dim))
# model.encoder.surf_token_embeds.bias["siconc"] = nn.Parameter(torch.zeros(model.encoder.surf_token_embeds.embed_dim))
old_bias = model.encoder.surf_token_embeds.bias
new_bias = torch.zeros(old_bias.shape)
new_bias[:4] = old_bias[:4]
model.encoder.surf_token_embeds.bias = nn.Parameter(new_bias)
# new_surf_head_weight = torch.zeros((16, 6, 512))
# new_surf_head_weight[:, :4, :] = model.encoder.surf_token_embeds.weights.reshape(16, 4, 512)
# model.encoder.surf_token_embeds.weights = new_surf_head_weight.reshape(-1, 512)

# new_surf_head_bias = torch.zeros((16, 6))
# new_surf_head_bias[:, :4] = pretrained_weights['net.decoder.surf_head.bias'].reshape(16, 4)
# pretrained_weights['net.decoder.surf_head.bias'] = new_surf_head_bias.reshape(-1)
# # Still needed?
# new_pretrained_weights = {}
# for key, value in pretrained_weights.items():
#     new_pretrained_weights[key[4:]] = value
# model.load_state_dict(new_pretrained_weights)

model.configure_activation_checkpointing()
model.train()
model = model.to(device)
over_size = 1 / (720 * 1440)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler()
writer = SummaryWriter(log_dir="runs/sst_finetune")

global_step = 0
for epoch in range(4):
    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
        input_batch: Batch
        target_batch: Batch
        opt.zero_grad()
        prediction: Batch = model(input_batch.to("cuda"))
        loss_value: torch.Tensor = loss(prediction, target_batch, over_size)
        scaler.scale(loss_value).backward()
        scaler.step(opt)
        scaler.update()

        writer.add_scalar("train/loss_step", loss_value.item(), global_step)

        if batch_idx == 0:
            # Log the first sample prediction/target as images for a quick qualitative check.
            writer.add_image(
                f"train/pred_sst_{epoch}",
                prediction.to("cpu").surf_vars["sst"][-1],
                global_step,
                dataformats="CHW",
            )
            writer.add_image(
                f"train/target_sst_{epoch}",
                target_batch.to("cpu").surf_vars["sst"][-1],
                global_step,
                dataformats="CHW",
            )
    print(f"Epoch {epoch+1} | loss={loss_value.item():.4f}")
    writer.add_scalar("train/loss_epoch", loss_value.item(), epoch)
    writer.flush()
    torch.save(model.state_dict(), f"sst_finetuned_{epoch}.ckpt")
    if epoch == 2:
        model.use_lora(True)

writer.close()
