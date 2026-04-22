import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from physicsnemo.models.unet import UNet


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def read_point_array(grid, name):
    arr = grid.GetPointData().GetArray(name)
    if arr is None:
        raise KeyError(f"Array '{name}' not found in VTU file")
    out = vtk_to_numpy(arr)
    if out.ndim == 1:
        out = out[:, None]
    return out.astype(np.float32)


def load_vtu_features(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    if grid is None or grid.GetNumberOfPoints() == 0:
        raise ValueError(f"Could not read points from {path}")

    u = read_point_array(grid, "U")  # N x 3
    t = read_point_array(grid, "T")  # N x 1
    p = read_point_array(grid, "p")  # N x 1
    wall = read_point_array(grid, "wallDistance")  # N x 1
    mask = read_point_array(grid, "vtkValidPointMask")  # N x 1

    x = np.concatenate([u, t, p, wall, mask], axis=1)  # N x 7
    return torch.from_numpy(x)


class VTUDataset(Dataset):
    def __init__(self, files):
        self.files = list(files)
        if len(self.files) == 0:
            raise ValueError("No VTU files were provided")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = load_vtu_features(self.files[idx])
        return {"x": x, "file": str(self.files[idx])}


def build_positional_embedding(nx=960, ny=96, nz=80):
    x = np.linspace(-1, 1, nx, dtype=np.float32)
    y = np.linspace(-1, 1, ny, dtype=np.float32)
    z = np.linspace(-1, 1, nz, dtype=np.float32)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    x_freq_sin = np.sin(xv * 72 * np.pi / 2)
    x_freq_cos = np.cos(xv * 72 * np.pi / 2)
    y_freq_sin = np.sin(yv * 8 * np.pi / 2)
    y_freq_cos = np.cos(yv * 8 * np.pi / 2)
    z_freq_sin = np.sin(zv * 8 * np.pi / 2)
    z_freq_cos = np.cos(zv * 8 * np.pi / 2)
    pos_embed = np.stack(
        (xv, x_freq_sin, x_freq_cos, yv, y_freq_sin, y_freq_cos, zv, z_freq_sin, z_freq_cos),
        axis=0,
    )
    return torch.from_numpy(pos_embed.astype(np.float32))


def save_checkpoint_simple(path, model, optimizer, scheduler, epoch):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(ckpt, os.path.join(path, f"epoch_{epoch}.pt"))


@torch.no_grad()
def validation_step(model, dataloader, pos_embed_base, epoch_or_step, plotting=False, device="cpu", name="default"):
    loss_epoch = 0.0
    num_samples = 0.0
    nx, ny, nz = 960, 96, 80

    model.eval()
    for i, data in enumerate(dataloader):
        x = data["x"].to(device)
        bs, _, chans = x.shape
        var = reshape_fortran(x, (bs, nx, ny, nz, chans))

        mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
        invar = torch.permute(var[..., 5:6], (0, 4, 1, 2, 3))
        pos_embed_tensor = pos_embed_base.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
        invar = torch.cat((invar, pos_embed_tensor), axis=1)
        outvar = torch.permute(var[..., 0:5], (0, 4, 1, 2, 3))

        pred_outvar = model(invar)
        outvar = outvar * mask
        pred_outvar = pred_outvar * mask
        loss_epoch += F.mse_loss(outvar, pred_outvar).item()
        num_samples += bs

        if plotting and i == 0:
            sample_idx = 0
            for chan in range(outvar.size(1)):
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                true_slice = outvar[sample_idx, chan, :, :, nz // 2].detach().cpu().numpy()
                pred_slice = pred_outvar[sample_idx, chan, :, :, nz // 2].detach().cpu().numpy()
                diff_slice = pred_slice - true_slice
                vmin, vmax = np.min(true_slice), np.max(true_slice)

                im = ax[0].imshow(true_slice, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax[0])
                im = ax[1].imshow(pred_slice, vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax[1])
                im = ax[2].imshow(diff_slice)
                fig.colorbar(im, ax=ax[2])

                ax[0].set_title("True")
                ax[1].set_title("Pred")
                ax[2].set_title("Diff")
                for a in ax:
                    a.set_aspect("equal")
                fig.tight_layout()
                plt.savefig(f"chan_{chan}_epoch_{epoch_or_step}_mid_z_slice_{name}.png", dpi=120)
                plt.close(fig)

    return loss_epoch / max(num_samples, 1.0)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")
    print(f"Using device: {device}")

    nx, ny, nz = 960, 96, 80
    n_points_expected = nx * ny * nz

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), betas=(0.9, 0.999), lr=cfg.start_lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_scheduler_gamma)

    # folder containing VTU files
    base_dir = Path("/mnt/c/Users/iaziz6/Downloads/Training")
    all_files = sorted(base_dir.glob("*.vtu"))
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .vtu files found in {base_dir}")

    requested_train = int(cfg.train_num_samples)
    requested_val = int(cfg.val_num_samples)
    if requested_train + requested_val > len(all_files):
        raise ValueError(
            f"Requested train+val={requested_train + requested_val} but only found {len(all_files)} VTU files"
        )

    train_files = all_files[:requested_train]
    val_files = all_files[requested_train:requested_train + requested_val]
    if len(val_files) == 0:
        raise ValueError("Validation file list is empty")

    print(f"Found {len(all_files)} VTU files")
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    # sanity check first file size
    first_x = load_vtu_features(train_files[0])
    if first_x.shape[0] != n_points_expected or first_x.shape[1] != 7:
        raise ValueError(
            f"Expected [N,7] with N={n_points_expected}; got {tuple(first_x.shape)} from {train_files[0].name}"
        )

    train_dataset = VTUDataset(train_files)
    val_dataset = VTUDataset(val_files)
    plot_dataset = VTUDataset(train_files[:1])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False, num_workers=0)
    plot_loader = DataLoader(plot_dataset, batch_size=1, shuffle=False, num_workers=0)

    pos_embed_base = build_positional_embedding(nx, ny, nz).to(device)

    train_losses, val_losses = [], []
    step_log, val_step_log = [], []
    log_every = int(cfg.get("log_every_steps", 5))
    val_every = int(cfg.get("val_every_steps", 50))
    global_step = 0

    def _save_loss_curve():
        fig, ax = plt.subplots()
        ax.plot(step_log, train_losses, label="Train", linewidth=1, marker=".", markersize=4)
        if val_step_log:
            ax.plot(val_step_log, val_losses, label="Val", marker="o", markersize=4)
        if step_log:
            ax.set_xlim(left=0, right=max(step_log) * 1.05 + 1)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title("Training Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig("loss_curve.png", dpi=120)
        plt.close(fig)

    max_epochs = int(cfg.max_epochs)
    for epoch in range(1, max_epochs + 1):
        model.train()
        for step_in_epoch, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            x = data["x"].to(device)
            bs, _, chans = x.shape
            var = reshape_fortran(x, (bs, nx, ny, nz, chans))

            mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
            invar = torch.permute(var[..., 5:6], (0, 4, 1, 2, 3))
            pos_embed_tensor = pos_embed_base.unsqueeze(0).repeat(bs, 1, 1, 1, 1)
            invar = torch.cat((invar, pos_embed_tensor), axis=1)
            outvar = torch.permute(var[..., 0:5], (0, 4, 1, 2, 3))

            pred_outvar = model(invar)
            outvar = outvar * mask
            pred_outvar = pred_outvar * mask
            loss = F.mse_loss(outvar, pred_outvar)
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            print(
                f"Epoch {epoch}/{max_epochs} | Step {step_in_epoch}/{len(train_loader)} | "
                f"Global {global_step} | Loss {loss.item():.6f} | LR {optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

            if global_step % log_every == 0:
                train_losses.append(loss.item())
                step_log.append(global_step)
                _save_loss_curve()

            if global_step % val_every == 0:
                val_loss = validation_step(
                    model, val_loader, pos_embed_base, global_step, plotting=True, device=device, name=f"val_step{global_step}"
                )
                _ = validation_step(
                    model, plot_loader, pos_embed_base, global_step, plotting=True, device=device, name=f"train_step{global_step}"
                )
                val_losses.append(val_loss)
                val_step_log.append(global_step)
                _save_loss_curve()
                model.train()

        if epoch % 2 == 0:
            save_checkpoint_simple("./checkpoints", model, optimizer, scheduler, epoch)


if __name__ == "__main__":
    main()
