import os
os.environ["ACCELERATE_USE_FIND_UNUSED_PARAMETERS"] = "true"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory

from src import utils
from src.loader import get_dataloader
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.models.lightunetr import LightUNETR
from src.utils import Logger, same_seeds
from accelerate import DistributedDataParallelKwargs

def train_one_epoch_gradaccumulate(
    model: torch.nn.Module,
    config: EasyDict,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    gradient_accumulation_steps = config.trainer.get("gradient_accumulation_steps", 1)
    # train
    model.train()
    for i, image_batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        logits = model(image_batch["image"])

        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            # 累积总损失（缩放后）
            total_loss += alpth * (loss / gradient_accumulation_steps)
            accelerator.log({"Train/" + name: float(loss)}, step=step)
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

        accelerator.backward(total_loss)
        # 梯度累积逻辑
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
            accelerator.log(
                {
                    "Train/Total Loss": float(total_loss * gradient_accumulation_steps),
                },
                step=step,
            )
            accelerator.print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss * gradient_accumulation_steps:1.5f} {log}",
                flush=True,
            )
            step += 1
        else:
            accelerator.log(
                {
                    "Train/Total Loss (unscaled)": float(total_loss * gradient_accumulation_steps),
                },
                step=step,
            )
            accelerator.print(
                f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss (accumulating): {total_loss * gradient_accumulation_steps:1.5f} {log}",
                flush=True,
            )
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
                f"Train/liver {metric_name}": float(batch_acc[0]),
                f"Train/kidney {metric_name}": float(batch_acc[1]),
                f"Train/spleen {metric_name}": float(batch_acc[2]),
                f"Train/pancreas {metric_name}": float(batch_acc[3]),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return step


def train_one_epoch(
    model: torch.nn.Module,
    config: EasyDict,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
    step: int,
):
    # train
    model.train()
    for i, image_batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        logits = model(image_batch["image"])

        total_loss = 0
        log = ""
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Train/" + name: float(loss)}, step=step)
            total_loss += alpth * loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log(
            {
                "Train/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{len(train_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update(
            {
                f"Train/mean {metric_name}": float(batch_acc.mean()),
                f"Train/liver {metric_name}": float(batch_acc[0]),
                f"Train/kidney {metric_name}": float(batch_acc[1]),
                f"Train/spleen {metric_name}": float(batch_acc[2]),
                f"Train/pancreas {metric_name}": float(batch_acc[3]),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return step


@torch.no_grad()
def val_one_epoch(
    model: torch.nn.Module,
    loss_functions: Dict[str, torch.nn.modules.loss._Loss],
    inference: monai.inferers.Inferer,
    val_loader: torch.utils.data.DataLoader,
    config: EasyDict,
    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
    step: int,
    post_trans: monai.transforms.Compose,
    accelerator: Accelerator,
    epoch: int,
):
    # val
    model.eval()
    for i, image_batch in enumerate(val_loader):
        logits = inference(image_batch["image"], model)
        total_loss = 0
        log = ""
        for name in loss_functions:
            loss = loss_functions[name](logits, image_batch["label"])
            accelerator.log({"Val/" + name: float(loss)}, step=step)
            log += f" {name} {float(loss):1.5f} "
            total_loss += loss
        val_outputs = [post_trans(i) for i in logits]
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch["label"])
        accelerator.log(
            {
                "Val/Total Loss": float(total_loss),
            },
            step=step,
        )
        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation [{i + 1}/{len(val_loader)}] Loss: {total_loss:1.5f} {log}",
            flush=True,
        )
        step += 1

    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()
        if accelerator.num_processes > 1:
            batch_acc = (
                accelerator.reduce(batch_acc.to(accelerator.device))
                / accelerator.num_processes
            )
        metrics[metric_name].reset()
        metric.update(
            {
                f"Val/mean {metric_name}": float(batch_acc.mean()),
                f"Val/liver {metric_name}": float(batch_acc[0]),
                f"Val/kidney {metric_name}": float(batch_acc[1]),
                f"Val/spleen {metric_name}": float(batch_acc[2]),
                f"Val/pancreas {metric_name}": float(batch_acc[3]),
            }
        )
    accelerator.print(
        f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation metric {metric}"
    )
    accelerator.log(metric, step=epoch)
    return (
        torch.Tensor([metric["Val/mean dice_metric"]]).to(accelerator.device),
        batch_acc,
        step,
    )


if __name__ == "__main__":
    save_dir = "/data_hdd/xyliu/lightunetr_experiments"
    # load yml
    config = EasyDict(
        yaml.load(open("config_abdomen_lightunetr.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    if config.is_abdomenct1k == True:
        config = config.abdomenct1k
        data_flag = "abdomenct1k"
    else:
        raise ValueError("Please set the correct config flag for the dataset you are using.")

    same_seeds(50)


    logging_dir = os.path.join(save_dir, "taskabdomen_lightunetr", "logs", config.finetune.checkpoint, str(datetime.now().strftime('%Y%m%d_%H%M%S')))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator()
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir, 
        # mixed_precision="bf16",  # 启用 BF16 混合精度
        kwargs_handlers=[ddp_kwargs]
    )
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print("Load Model...")
    model = LightUNETR(
        in_channels= 1,
        out_channels= 4,
        embedding_dim= 27
    )

    image_size = config.trainer.image_size

    accelerator.print(model)
    accelerator.print("Load Dataloader...")
    train_loader, val_loader, train_images, train_transform = get_dataloader(config, data_flag)

    inference = monai.inferers.SlidingWindowInferer(
        roi_size=(96,96,96),
        overlap=0.5,
        sw_device=accelerator.device,
        device=accelerator.device,
    )
    metrics = {
        "dice_metric": monai.metrics.DiceMetric(
            include_background=True,
            reduction=monai.utils.MetricReduction.MEAN_BATCH,
            get_not_nans=False,
        ),
        # 'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=True, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=False)
    }
    post_trans = monai.transforms.Compose(
        [
            monai.transforms.Activations(sigmoid=True),
            monai.transforms.AsDiscrete(threshold=0.5),
        ]
    )

    optimizer = optim_factory.create_optimizer_v2(
        model,
        opt=config.trainer.optimizer,
        weight_decay=config.trainer.weight_decay,
        lr=config.trainer.lr,
        betas=(0.9, 0.95),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.trainer.warmup,
        max_epochs=config.trainer.num_epochs,
    )
    loss_functions = {
        "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
        "dice_loss": monai.losses.DiceLoss(
            smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
        ),
    }

    step = 0
    best_eopch = -1
    val_step = 0
    starting_epoch = 0
    best_acc = 0
    best_class = []

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )

    # resume training
    if config.trainer.resume:
        model, starting_epoch, step, val_step = utils.resume_train_state(
            model, "{}".format(config.finetune.checkpoint), train_loader, accelerator
        )

    # Start Training
    accelerator.print("Start Training ... ")
    mean_acc = torch.Tensor([0]).to(accelerator.device)
    batch_acc = torch.Tensor([0]).to(accelerator.device)
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # train
        step = train_one_epoch(
            model,
            config,
            loss_functions,
            train_loader,
            optimizer,
            scheduler,
            metrics,
            post_trans,
            accelerator,
            epoch,
            step,
        )
        # val every 20 epochs
        if (epoch + 1) % 20 == 0 or epoch == config.trainer.num_epochs - 1 or epoch == 0:
            accelerator.print(f"Start Validation at epoch {epoch + 1} ...")
            mean_acc, batch_acc, val_step = val_one_epoch(
                model,
                loss_functions,
                inference,
                val_loader,
                config,
                metrics,
                val_step,
                post_trans,
                accelerator,
                epoch,
            )

        accelerator.print(
            f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] lr = {scheduler.get_last_lr()} best acc: {best_acc}, mean acc: {mean_acc}, mean class: {batch_acc}"
        )

        # save model
        if mean_acc > best_acc:
            accelerator.save_state(
                output_dir=f"{save_dir}/taskabdomen_lightunetr/model_store/{config.finetune.checkpoint}/best"
            )
            best_acc = mean_acc
            best_class = batch_acc
            best_epoch = epoch
        if epoch % 50 == 0:
            accelerator.save_state(
                output_dir=f"{save_dir}/taskabdomen_lightunetr/model_store/{config.finetune.checkpoint}/epoch{epoch+1}"
            )

    # save final model
    accelerator.save_state(
        output_dir=f"{save_dir}/taskabdomen_lightunetr/model_store/{config.finetune.checkpoint}/final"
    )
    accelerator.print(f"best epoch: {best_epoch}")

    accelerator.print(f"best dice mean acc: {best_acc}")
    accelerator.print(f"best dice accs: {best_class}")
    sys.exit(1)
