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

MAE = nn.L1Loss()

# surf_weights = {"MSL": 1.5, "U10": 0.77, "V10": 0.66, "2T": 3.0}

# atmos_weights = {"z": 2.8, "Q": 0.78, "T": 1.7, "U": 0.87, "V": 0.6}


model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "sst", "ci"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"),
    autocast=True,
)

# normalization (yearly) means
locations["sst"] = 17.1771
locations["ci"] = 0.1589

# normalization (yearly) standard deviations
scales["sst"] = 10.301042
scales["ci"] = 0.32129946


def sensible_heat(
    sst: torch.Tensor,
    t2m: torch.Tensor,
    wind_speed: torch.Tensor,
    rhoao: float = 1.2250,  # density at 15C
    cp: float = 1.006e3,  # 1.006kJ
    cs: float = 1e-3,
    # cs: float = 0.9e-3,
):
    return -rhoao * cp * cs * wind_speed * (sst - t2m)


def loss(
    pred: Batch,
    target: Batch,
    ice_cover: torch.Tensor,
    over_size: int,
    alpha: float = 0.25,
    beta: float = 1.0,
) -> torch.Tensor:
    surf_values = pred.surf_vars.values()
    pred_sst = surf_values["sst"]
    pred_t2m = surf_values["t2m"]
    pred_u10 = surf_values["u10"]
    pred_v10 = surf_values["v10"]
    pred_wind_speed = torch.sqrt(pred_u10**2, pred_v10**2)

    target_sst = target["sst"]
    target_t2m = target["t2m"]
    target_u10 = target["u10"]
    target_v10 = target["v10"]
    target_wind_speed = torch.sqrt(target_u10**2, target_v10**2)

    pred_sh = sensible_heat(pred_sst, pred_t2m, pred_wind_speed)
    target_sh = sensible_heat(target_sst, target_t2m, target_wind_speed)

    norm_pred_sst = normalise_surf_var(pred_sst, "sst", model.surf_stats)
    norm_target_sst = normalise_surf_var(target_sst, "sst", model.surf_stats)
    phys_loss_pre = abs(norm_pred_sst - norm_target_sst) * abs(pred_sh - target_sh)
    # TODO is this the correct interpretation
    phys_loss_pre = torch.where(ice_cover > 0.3, torch.nan, phys_loss_pre)
    phys_loss = torch.nansum(phys_loss_pre) * over_size

    mae_loss = nn.functional.mse_loss(norm_pred_sst, target_sst)

    return mae_loss + phys_loss


if not torch.cuda.is_available():
    raise RuntimeError("Need CUDA for Aurora fine-tuning.")
device = torch.device("cuda")
data_path = Path("./data/downloads")

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_batches)

model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.configure_activation_checkpointing()
model.train()
model = model.to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
writer = SummaryWriter(log_dir="runs/sst_finetune")

global_step = 0
for epoch in range(2):
    for batch_idx, (batch, target) in enumerate(dataloader):
        opt.zero_grad()
        prediction = model(batch.to("cuda"))
        loss_value: torch.Tensor = loss(prediction, target, 0)
        loss_value.backward()
        opt.step()

        writer.add_scalar("train/loss_step", loss_value.item(), global_step)

        if batch_idx == 0:
            # Log the first sample prediction/target as images for a quick qualitative check.
            writer.add_image(
                "train/pred_ishf",
                prediction[0].unsqueeze(0),
                global_step,
                dataformats="CHW",
            )
            writer.add_image(
                "train/target_ishf",
                target[0].unsqueeze(0),
                global_step,
                dataformats="CHW",
            )
    print(f"Epoch {epoch+1} | loss={loss_value.item():.4f}")
    writer.add_scalar("train/loss_epoch", loss_value.item(), epoch)
    writer.flush()
    torch.save(model.state_dict(), f"sst_finetuned_{epoch}.ckpt")
writer.close()
