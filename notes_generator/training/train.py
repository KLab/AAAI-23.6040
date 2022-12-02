import math
import shutil
import typing
from logging import getLogger
from pathlib import Path

import mlflow
import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping, ModelCheckpoint
from ignite.metrics import Average
from torch.nn.utils import clip_grad_norm_

from notes_generator.training.evaluate import evaluate

logger = getLogger(__name__)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def write_metrics(metrics, writer, mode: str, epoch: int, disable_mlflow: bool):
    loss = metrics["loss"]
    logger.info(f"{mode} Results - Epoch: {epoch}  " f"Avg loss: {loss:.4f}")
    # tensorboard
    writer.add_scalar(f"{mode}-avg_loss", loss, epoch)
    writer.add_scalar(f"{mode}-avg_loss_onset", metrics["loss-onset"], epoch)
    if "loss-notes" in metrics:
        writer.add_scalar(f"{mode}-avg_loss_notes", metrics["loss-notes"], epoch)
    # mlflow
    if disable_mlflow:
        return
    mlflow.log_metric(f"{mode}-avg_loss", loss, epoch)
    mlflow.log_metric(f"{mode}-avg_loss_onset", metrics["loss-onset"], epoch)
    if "loss-notes" in metrics:
        mlflow.log_metric(f"{mode}-avg_loss_notes", metrics["loss-notes"], epoch)


def score_function(engine):
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def train_ignite(
    epochs,
    model,
    log_dir,
    batch_size,
    train_loader,
    valid_loader,
    optimizer,
    lr_scheduler,
    writer,
    device,
    onset_only: bool = True,
    fuzzy_width: int = 1,
    fuzzy_scale: float = 1.0,
    merge_scale: typing.Optional[float] = None,
    patience: int = 10,
    enable_early_stop: bool = True,
    disable_eval: bool = False,
    resume_checkpoint: int = None,
    lr_find: bool = False,
    epoch_length=100,
    start_lr=1e-7,
    end_lr=1e-1,
    clip_gradient_norm: float = 3,
    loss_interval: int = 100,
    validation_interval: int = 200,
    checkpoint_interval: int = 200,
    n_saved_checkpoint: int = 10,
    n_saved_model: int = 40,
    disable_mlflow: bool = False,
    warmup_steps: int = 0,
    send_model: bool = False,
    eval_tolerance: float = 0.05,
):
    if lr_find:
        lr_find_loss = []
        lr_find_lr = []
        optimizer = torch.optim.Adam(model.parameters(), start_lr)
        lr_find_epochs = 2
        lr_lambda = lambda x: math.exp(
            x * math.log(end_lr / start_lr) / (lr_find_epochs * epoch_length)
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        epochs = lr_find_epochs
        smoothing = 0.05

    def update_model(engine, batch):
        if warmup_steps > 0 and engine.state.iteration < warmup_steps:
            lr_scale = min(1.0, float(engine.state.iteration + 1) / float(warmup_steps))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * target_lr
        optimizer.zero_grad()
        predictions, losses = model.run_on_batch(batch, fuzzy_width, fuzzy_scale, merge_scale)
        loss = sum(losses.values())
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_gradient_norm)
        optimizer.step()
        if lr_scheduler:
            if warmup_steps > 0 and engine.state.iteration < warmup_steps:
                pass
            else:
                lr_scheduler.step()
        if lr_find:
            loss_v = loss.item()
            i = engine.state.iteration
            lr_step = optimizer.state_dict()["param_groups"][0]["lr"]
            lr_find_lr.append(lr_step)
            if i == 1:
                lr_find_loss.append(loss_v)
            else:
                loss_v = smoothing * loss_v + (1 - smoothing) * lr_find_loss[-1]
                lr_find_loss.append(loss_v)
        losses = {key: value.item() for key, value in {"loss": loss, **losses}.items()}
        i = engine.state.iteration
        for key, value in losses.items():
            # tensorboard
            writer.add_scalar(key, value, global_step=i)
            # mlflow
            if not disable_mlflow:
                mlflow.log_metric(key, value, step=i)
        return predictions, losses

    def evaluate_func(engine, batch):
        model.eval()
        with torch.no_grad():
            predictions, losses = model.run_on_batch(batch)
            loss = sum(losses.values())
            losses = {key: value.item() for key, value in {"loss": loss, **losses}.items()}
            model.train()
            return predictions, losses

    trainer = Engine(update_model)
    evaluator = Engine(evaluate_func)
    target_lr = [pg["lr"] for pg in optimizer.param_groups][-1]

    checkpoint = Path(log_dir) / "checkpoint"

    @trainer.on(Events.STARTED)
    def resume_training(engine):
        if resume_checkpoint:
            engine.state.iteration = resume_checkpoint
            engine.state.epoch = int(resume_checkpoint / engine.state.epoch_length)

    @trainer.on(Events.COMPLETED)
    def log_model(engine):
        if not send_model or disable_mlflow:
            return
        mlflow.pytorch.log_model(model, "model")
        for model_path in sorted(checkpoint.rglob("model*")):
            step = str(model_path).split(".")[0].split("_")[-1]
            model.load_state_dict(torch.load(checkpoint / model_path))
            mlflow.pytorch.log_model(model, f"model_{step}")

    @trainer.on(Events.COMPLETED)
    def write_lr_find(engine):
        if not lr_find:
            return
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.plot(lr_find_lr, lr_find_loss)
        plt.xscale("log")
        plt.show()

    @trainer.on(Events.ITERATION_COMPLETED(every=loss_interval))
    def log_training_loss(engine):
        loss = engine.state.output[1]["loss"]
        iteration_max = engine.state.max_epochs * engine.state.epoch_length
        logger.info(f"Iteration[{engine.state.iteration}/{iteration_max}] " f"Loss: {loss:.4f}")

    @trainer.on(Events.ITERATION_COMPLETED(every=validation_interval))
    def log_validation_results(engine):
        i = engine.state.iteration
        lr = [pg["lr"] for pg in optimizer.param_groups][-1]
        writer.add_scalar("learning_rate", lr, global_step=i)
        mlflow.log_metric("learning_rate", lr, step=i)
        evaluator.run(cycle(valid_loader), epoch_length=epoch_length_valid)
        model.eval()
        with torch.no_grad():
            if disable_eval:
                pass
            else:
                for key, value in evaluate(model, valid_loader, device, eval_tolerance).items():
                    k = "validation-" + key.replace(" ", "_")
                    v = np.mean(value)
                    # tensorboard
                    writer.add_scalar(k, v, global_step=i)
                    # mlflow
                    if not disable_mlflow:
                        mlflow.log_metric(k, v, step=i)
        metrics = evaluator.state.metrics
        write_metrics(metrics, writer, "validation", engine.state.epoch, disable_mlflow)
        model.train()

    avg_loss = Average(output_transform=lambda output: output[1]["loss"])
    avg_loss_onset = Average(output_transform=lambda output: output[1]["loss-onset"])
    if not onset_only:
        avg_loss_notes = Average(output_transform=lambda output: output[1]["loss-notes"])
        avg_loss_notes.attach(evaluator, "loss-notes")
    avg_loss.attach(trainer, "loss")
    avg_loss.attach(evaluator, "loss")
    avg_loss_onset.attach(evaluator, "loss-onset")
    if enable_early_stop:
        handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)
    to_save = {"trainer": trainer, "optimizer": optimizer}
    if lr_scheduler:
        to_save["lr_scheduler"] = lr_scheduler
    if checkpoint.exists() and not resume_checkpoint:
        shutil.rmtree(str(checkpoint))
    handler = Checkpoint(
        to_save,
        DiskSaver(str(checkpoint), create_dir=True, require_empty=False),
        n_saved=n_saved_checkpoint,
    )
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=checkpoint_interval), handler)
    model_handler = ModelCheckpoint(
        dirname=str(checkpoint),
        filename_prefix="model",
        n_saved=n_saved_model,
        create_dir=True,
        require_empty=False,
    )
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=checkpoint_interval), model_handler, {"mymodel": model}
    )
    epoch_length_valid = len(list(valid_loader))
    if resume_checkpoint:
        check_point_path = f"{str(checkpoint)}/checkpoint_{resume_checkpoint}.pth"
        model_state_path = f"{str(checkpoint)}/model_mymodel_{resume_checkpoint}.pth"
        to_load = {"trainer": trainer, "optimizer": optimizer, "lr_scheduler": lr_scheduler}
        checkpoint_ = torch.load(check_point_path, map_location=torch.device(device))
        Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint_)
        model_state = torch.load(model_state_path, map_location=torch.device(device))
        model.load_state_dict(model_state)
        # release memory
        del model_state
        del checkpoint_
    logger.info(f"epoch_length: {epoch_length} epoch_length_valid: {epoch_length_valid}")
    trainer.run(cycle(train_loader), max_epochs=epochs, epoch_length=epoch_length)
