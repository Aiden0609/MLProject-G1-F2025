from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from aurora.model.aurora_lite import AuroraLite
from aurora.model.decoder_lite import MLPDecoderLite

from data_prep import AuroraLiteFluxDataset


def collate_batches(batch_items):
    """Stack (Batch, target) pairs into a batched Batch and tensor targets."""
    batches, targets = zip(*batch_items)

    surf_vars = {
        k: torch.cat([b.surf_vars[k] for b in batches], dim=0) for k in batches[0].surf_vars
    }
    atmos_vars = {
        k: torch.cat([b.atmos_vars[k] for b in batches], dim=0) for k in batches[0].atmos_vars
    }

    static_vars = batches[0].static_vars  # Same for every sample.

    metadata = batches[0].metadata
    metadata = type(metadata)(
        lat=metadata.lat,
        lon=metadata.lon,
        time=tuple(b.metadata.time[0] for b in batches),
        atmos_levels=metadata.atmos_levels,
    )

    stacked_batch = type(batches[0])(
        surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars, metadata=metadata
    )

    targets = torch.cat(targets, dim=0)  # (B, H, W)

    return stacked_batch, targets


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Need CUDA for Aurora fine-tuning.")
    device = torch.device("cuda")

    data_path = Path("./data/downloads")
    dataset = AuroraLiteFluxDataset(data_path, history=2, target_var="ishf")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batches)

    modelAurora = AuroraLite(
        use_lora=False,
        autocast=True,
        surf_vars=("2t", "10u", "10v", "msl"),
        static_vars=("lsm", "z", "slt"),
        atmos_vars=("z", "u", "v", "t", "q"),
    )
    modelAurora.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
    modelAurora = modelAurora.to(device)
    modelAurora.eval()

    modelDecoder = MLPDecoderLite(
        surf_vars_new=("ishf",),
        patch_size=modelAurora.decoder.patch_size,
        embed_dim=2 * modelAurora.encoder.embed_dim,
        hidden_dims=[512, 512, 256],
    )
    checkpoint = torch.load("./lite-decoder.ckpt", map_location="cpu")
    modelDecoder.load_state_dict(checkpoint, strict=False)
    modelDecoder = modelDecoder.to(device)
    modelDecoder.train()

    opt = torch.optim.AdamW(modelDecoder.parameters(), lr=3e-4)

    writer = SummaryWriter(log_dir="runs/ishf_finetune")
    global_step = 0

    for epoch in range(10):
        for batch_idx, (batch, target) in enumerate(dataloader):
            with torch.inference_mode():
                _, latent = modelAurora.forward(batch)
                latent_decoder = latent.detach().clone()

            opt.zero_grad()
            preds_new = modelDecoder(latent_decoder, batch.metadata.lat, batch.metadata.lon)
            pred_flux = preds_new["ishf"].squeeze(1)  # (B, H, W)
            loss_value = F.mse_loss(pred_flux, target.to(device))
            loss_value.backward()
            opt.step()

            writer.add_scalar("train/loss_step", loss_value.item(), global_step)

            if batch_idx == 0:
                # Log the first sample prediction/target as images for a quick qualitative check.
                writer.add_image(
                    "train/pred_ishf",
                    pred_flux[0].unsqueeze(0),
                    global_step,
                    dataformats="CHW",
                )
                writer.add_image(
                    "train/target_ishf",
                    target[0].unsqueeze(0),
                    global_step,
                    dataformats="CHW",
                )

            global_step += 1

        print(f"Epoch {epoch+1} | loss={loss_value.item():.4f}")
        writer.add_scalar("train/loss_epoch", loss_value.item(), epoch)
        writer.flush()

        torch.save(modelDecoder.state_dict(), f"ishf_decoder_finetuned_{epoch}.ckpt")

    writer.close()


if __name__ == "__main__":
    main()
