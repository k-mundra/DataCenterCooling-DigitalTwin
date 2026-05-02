# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from physicsnemo.datapipes.cae.mesh_datapipe import MeshDatapipe
from physicsnemo.distributed import DistributedManager
import vtk  # kept for environment parity with the original scripts
from physicsnemo.models.unet import UNet
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import hydra
import torch.nn.functional as F
from physicsnemo.launch.utils.checkpoint import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
import numpy as np


def dilate_mask_3d(mask: torch.Tensor, padding_size: int) -> torch.Tensor:
    inverted_mask = (~mask.bool()).float()
    kernel_size = 2 * padding_size + 1
    kernel = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    dilated_result = torch.clamp(F.conv3d(inverted_mask, kernel, padding=padding_size), 0, 1)
    dilated_result = (~dilated_result.bool()).float()
    return dilated_result


def reshape_fortran(x: torch.Tensor, shape) -> torch.Tensor:
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = (pred - target) ** 2 * mask
    denom = (mask.sum() * pred.shape[1]).clamp_min(1.0)
    return weighted.sum() / denom


def interior_mask(mask: torch.Tensor) -> torch.Tensor:
    m = mask.clone()
    m[:, :, 0, :, :] = 0.0
    m[:, :, -1, :, :] = 0.0
    m[:, :, :, 0, :] = 0.0
    m[:, :, :, -1, :] = 0.0
    m[:, :, :, :, 0] = 0.0
    m[:, :, :, :, -1] = 0.0
    return m


def ddx(f: torch.Tensor, dx: float) -> torch.Tensor:
    return (f[:, :, 2:, 1:-1, 1:-1] - f[:, :, :-2, 1:-1, 1:-1]) / (2.0 * dx)


def ddy(f: torch.Tensor, dy: float) -> torch.Tensor:
    return (f[:, :, 1:-1, 2:, 1:-1] - f[:, :, 1:-1, :-2, 1:-1]) / (2.0 * dy)


def ddz(f: torch.Tensor, dz: float) -> torch.Tensor:
    return (f[:, :, 1:-1, 1:-1, 2:] - f[:, :, 1:-1, 1:-1, :-2]) / (2.0 * dz)


def laplacian(f: torch.Tensor, dx: float, dy: float, dz: float) -> torch.Tensor:
    c = f[:, :, 1:-1, 1:-1, 1:-1]
    d2x = (f[:, :, 2:, 1:-1, 1:-1] - 2.0 * c + f[:, :, :-2, 1:-1, 1:-1]) / (dx * dx)
    d2y = (f[:, :, 1:-1, 2:, 1:-1] - 2.0 * c + f[:, :, 1:-1, :-2, 1:-1]) / (dy * dy)
    d2z = (f[:, :, 1:-1, 1:-1, 2:] - 2.0 * c + f[:, :, 1:-1, 1:-1, :-2]) / (dz * dz)
    return d2x + d2y + d2z


def compute_ns_residuals(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    p: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    nu: float = 0.01,
    rho: float = 1.0,
):
    ux, uy, uz = ddx(u, dx), ddy(u, dy), ddz(u, dz)
    vx, vy, vz = ddx(v, dx), ddy(v, dy), ddz(v, dz)
    wx, wy, wz = ddx(w, dx), ddy(w, dy), ddz(w, dz)

    px, py, pz = ddx(p, dx), ddy(p, dy), ddz(p, dz)

    uc = u[:, :, 1:-1, 1:-1, 1:-1]
    vc = v[:, :, 1:-1, 1:-1, 1:-1]
    wc = w[:, :, 1:-1, 1:-1, 1:-1]

    lap_u = laplacian(u, dx, dy, dz)
    lap_v = laplacian(v, dx, dy, dz)
    lap_w = laplacian(w, dx, dy, dz)

    continuity = ux + vy + wz
    momentum_x = uc * ux + vc * uy + wc * uz + (1.0 / rho) * px - nu * lap_u
    momentum_y = uc * vx + vc * vy + wc * vz + (1.0 / rho) * py - nu * lap_v
    momentum_z = uc * wx + vc * wy + wc * wz + (1.0 / rho) * pz - nu * lap_w

    return {
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "momentum_z": momentum_z,
    }


def temperature_residual_interior(
    T: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    dx: float,
    dy: float,
    dz: float,
    alpha: float,
    source_term: float = 0.0,
) -> torch.Tensor:
    Tc = T[:, :, 1:-1, 1:-1, 1:-1]
    uc = u[:, :, 1:-1, 1:-1, 1:-1]
    vc = v[:, :, 1:-1, 1:-1, 1:-1]
    wc = w[:, :, 1:-1, 1:-1, 1:-1]

    dTdx = ddx(T, dx)
    dTdy = ddy(T, dy)
    dTdz = ddz(T, dz)
    lap_T = laplacian(T, dx, dy, dz)

    return uc * dTdx + vc * dTdy + wc * dTdz - alpha * lap_T - source_term


@torch.no_grad()
def validation_step(model, dataset, pos_embed_tensor, epoch, plotting=False, name="default"):
    loss_epoch = 0.0
    num_samples = 0.0

    nx, ny, nz = 960, 96, 80
    for i, data in enumerate(dataset):
        bs, _, chans = data[0]["x"].shape
        var = reshape_fortran(data[0]["x"], (bs, nx, ny, nz, chans))

        mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
        invar = torch.permute(var[..., 5:6], (0, 4, 1, 2, 3))
        invar = torch.cat((invar, pos_embed_tensor[:bs]), axis=1)
        outvar = torch.permute(var[..., 0:5], (0, 4, 1, 2, 3))
        pred_outvar = model(invar)

        loss_epoch += masked_mse(pred_outvar, outvar, mask)
        num_samples += invar.shape[0]

        if plotting and i == 0:
            for chan in range(outvar.size(1)):
                fig, ax = plt.subplots(1, 3)
                vmin = np.min(outvar[0, chan, :, :, nz // 2].detach().cpu().numpy())
                vmax = np.max(outvar[0, chan, :, :, nz // 2].detach().cpu().numpy())

                im = ax[0].imshow(outvar[0, chan, :, :, nz // 2].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax[0])
                im = ax[1].imshow(pred_outvar[0, chan, :, :, nz // 2].detach().cpu().numpy(), vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax[1])
                im = ax[2].imshow((pred_outvar[0, chan, :, :, nz // 2] - outvar[0, chan, :, :, nz // 2]).detach().cpu().numpy())
                fig.colorbar(im, ax=ax[2])

                ax[0].set_aspect("equal")
                ax[1].set_aspect("equal")
                ax[2].set_aspect("equal")
                ax[0].set_title("True")
                ax[1].set_title("Pred")
                ax[2].set_title("Diff")

                plt.savefig(f"chan_{chan}_epoch_{epoch}_mid_z_slice_{name}.png")
                plt.close()

    return loss_epoch.detach() / max(num_samples, 1.0)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_physics_informed")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")
    LaunchLogger.initialize()

    nx, ny, nz = 960, 96, 80

    train_num_samples = cfg.get("train_num_samples", 15)
    val_num_samples = cfg.get("val_num_samples", 15)
    train_batch_size = cfg.get("train_batch_size", 1)
    val_batch_size = cfg.get("val_batch_size", 1)
    max_epochs = cfg.get("max_epochs", 40)
    log_every = cfg.get("log_every_steps", 1)
    val_every = cfg.get("val_every_steps", 7)
    save_every = cfg.get("save_every_epochs", 10)
    num_workers = cfg.get("num_workers", 4)

    phy_wt = cfg.get("phy_wt", 1.0)
    temp_phy_wt = cfg.get("temp_phy_wt", phy_wt)
    alpha = cfg.get("alpha", 0.01)
    nu = cfg.get("nu", 0.01)
    rho = cfg.get("rho", 1.0)
    temp_source = cfg.get("temp_source", 0.0)
    mask_dilation = cfg.get("mask_dilation", 3)
    plot_validation_slices = cfg.get("plot_validation_slices", False)

    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)

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

    DistributedManager.initialize()
    dist = DistributedManager()

    max_batch = max(train_batch_size, val_batch_size)
    pos_embed_tensor = torch.from_numpy(pos_embed).to(torch.float32).to(dist.device)
    pos_embed_tensor = pos_embed_tensor.unsqueeze(0).repeat(max_batch, 1, 1, 1, 1)

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(dist.device)

    bounds = cfg.get("bounds", (0.0, 40.0, -3.95, 0.05, 0.0, 3.2))
    dx = (bounds[1] - bounds[0]) / max(nx - 1, 1)
    dy = (bounds[3] - bounds[2]) / max(ny - 1, 1)
    dz = (bounds[5] - bounds[4]) / max(nz - 1, 1)

    mean_dict = {
        "T": cfg.get("mean_T", 39.0),
        "U": cfg.get("mean_U", 1.5983600616455078),
        "p": cfg.get("mean_p", 6.1226935386657715),
        "wallDistance": cfg.get("mean_wallDistance", 0.6676982045173645),
    }
    std_dict = {
        "T": cfg.get("std_T", 4.0),
        "U": cfg.get("std_U", 1.3656059503555298),
        "p": cfg.get("std_p", 4.166020393371582),
        "wallDistance": cfg.get("std_wallDistance", 0.45233625173568726),
    }

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
        )

    data_path = cfg.get("data_path", "/home/hice1/iaziz6/Training/Training")
    val_data_path = cfg.get("val_data_path", data_path)

    data_dir = to_absolute_path(data_path)
    dataset = MeshDatapipe(
        data_dir=data_dir,
        file_format="vtu",
        variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
        num_variables=7,
        num_samples=train_num_samples,
        batch_size=train_batch_size,
        num_workers=num_workers,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        shuffle=True,
        parallel=False,
    )

    if dist.rank == 0:
        val_data_dir = to_absolute_path(val_data_path)
        val_dataset = MeshDatapipe(
            data_dir=val_data_dir,
            file_format="vtu",
            variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
            num_variables=7,
            num_samples=val_num_samples,
            batch_size=val_batch_size,
            num_workers=num_workers,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            shuffle=False,
            parallel=False,
        )

        train_dataset_plotting = MeshDatapipe(
            data_dir=data_dir,
            file_format="vtu",
            variables=["U", "T", "p", "wallDistance", "vtkValidPointMask"],
            num_variables=7,
            num_samples=1,
            batch_size=val_batch_size,
            num_workers=0,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            shuffle=False,
            parallel=False,
        )

    use_fused = bool(cfg.get("fused_adam", True)) and dist.device.type == "cuda"
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.start_lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        fused=use_fused,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_scheduler_gamma)

    checkpoint_dir = cfg.get("checkpoint_dir", "./checkpoints_physics_complete")
    loaded_epoch = load_checkpoint(
        checkpoint_dir,
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    train_total_losses, train_data_losses = [], []
    train_ns_losses, train_temp_losses = [], []
    val_losses = []
    step_log, val_step_log = [], []
    global_step = 0

    def _save_loss_curve() -> None:
        fig, ax = plt.subplots()
        if step_log:
            ax.plot(step_log, train_total_losses, label="Train Total", linewidth=1, marker=".", markersize=4)
            ax.plot(step_log, train_data_losses, label="Train Data", linewidth=1)
            ax.plot(step_log, train_ns_losses, label="Train NS", linewidth=1)
            ax.plot(step_log, train_temp_losses, label="Train Temp", linewidth=1)
        if val_step_log:
            ax.plot(val_step_log, val_losses, label="Val Data", marker="o", markersize=4)
        if step_log:
            ax.set_xlim(left=0, right=max(step_log) * 1.05 + 1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Physics-Informed Training Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig("loss_curve_complete.png", dpi=120)
        plt.close(fig)

    logger.info(
        f"Using physics-informed training with train_num_samples={train_num_samples}, "
        f"val_num_samples={val_num_samples}, max_epochs={max_epochs}, "
        f"log_every_steps={log_every}, val_every_steps={val_every}."
    )

    for epoch in range(max(1, loaded_epoch + 1), max_epochs + 1):
        with LaunchLogger("train", epoch=epoch, num_mini_batch=len(dataset), epoch_alert_freq=1) as log:
            for step_in_epoch, data in enumerate(dataset, 1):
                optimizer.zero_grad()
                bs, _, chans = data[0]["x"].shape
                var = reshape_fortran(data[0]["x"], (bs, nx, ny, nz, chans))

                mask = torch.permute(var[..., 6:7], (0, 4, 1, 2, 3))
                mask_phys = interior_mask(dilate_mask_3d(mask, mask_dilation))
                mask_int = mask_phys[:, :, 1:-1, 1:-1, 1:-1]

                invar = torch.permute(var[..., 5:6], (0, 4, 1, 2, 3))
                invar = torch.cat((invar, pos_embed_tensor[:bs]), axis=1)

                outvar = torch.permute(var[..., 0:5], (0, 4, 1, 2, 3))
                pred_outvar = model(invar)

                data_loss = masked_mse(pred_outvar, outvar, mask)

                u_phys = pred_outvar[:, 0:1] * std_dict["U"] + mean_dict["U"]
                v_phys = pred_outvar[:, 1:2] * std_dict["U"] + mean_dict["U"]
                w_phys = pred_outvar[:, 2:3] * std_dict["U"] + mean_dict["U"]
                T_phys = pred_outvar[:, 3:4] * std_dict["T"] + mean_dict["T"]
                p_phys = pred_outvar[:, 4:5] * std_dict["p"] + mean_dict["p"]

                ns_residuals = compute_ns_residuals(
                    u=u_phys,
                    v=v_phys,
                    w=w_phys,
                    p=p_phys,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    nu=nu,
                    rho=rho,
                )

                ns_mask_denom = mask_int.sum().clamp_min(1.0)
                ns_phy_loss = 0.0
                for residual in ns_residuals.values():
                    ns_phy_loss = ns_phy_loss + (mask_int * residual ** 2).sum() / ns_mask_denom

                temp_res = temperature_residual_interior(
                    T=T_phys,
                    u=u_phys,
                    v=v_phys,
                    w=w_phys,
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    alpha=alpha,
                    source_term=temp_source,
                )
                temp_phy_loss = (mask_int * temp_res ** 2).sum() / mask_int.sum().clamp_min(1.0)

                loss = data_loss + phy_wt * ns_phy_loss + temp_phy_wt * temp_phy_loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                log.log_minibatch({"Mini-batch total loss": loss.detach()})
                log.log_minibatch({"Mini-batch data loss": data_loss.detach()})
                log.log_minibatch({"Mini-batch NS phy loss": ns_phy_loss.detach()})
                log.log_minibatch({"Mini-batch T phy loss": temp_phy_loss.detach()})

                global_step += 1
                print(
                    f"Epoch {epoch}/{max_epochs} | "
                    f"Step {step_in_epoch}/{len(dataset)} | Global {global_step} | "
                    f"Total {loss.item():.6f} | Data {data_loss.item():.6f} | "
                    f"NS {ns_phy_loss.item():.6f} | T {temp_phy_loss.item():.6f} | "
                    f"LR {optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

                if dist.rank == 0 and global_step % log_every == 0:
                    step_log.append(global_step)
                    train_total_losses.append(loss.detach().item())
                    train_data_losses.append(data_loss.detach().item())
                    train_ns_losses.append(ns_phy_loss.detach().item())
                    train_temp_losses.append(temp_phy_loss.detach().item())
                    _save_loss_curve()

                if dist.rank == 0 and global_step % val_every == 0:
                    val_loss = validation_step(
                        model,
                        val_dataset,
                        pos_embed_tensor,
                        global_step,
                        plotting=plot_validation_slices,
                        name=f"val_step{global_step}",
                    )
                    if plot_validation_slices:
                        _ = validation_step(
                            model,
                            train_dataset_plotting,
                            pos_embed_tensor,
                            global_step,
                            plotting=True,
                            name=f"train_step{global_step}",
                        )
                    val_losses.append(val_loss.item())
                    val_step_log.append(global_step)
                    _save_loss_curve()

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.world_size > 1:
            torch.distributed.barrier()

        if epoch % save_every == 0 and dist.rank == 0:
            save_checkpoint(
                checkpoint_dir,
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
