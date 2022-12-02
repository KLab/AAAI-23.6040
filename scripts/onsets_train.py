import argparse
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import mlflow
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from notes_generator.constants import *
from notes_generator.models.onsets import SimpleOnsets
from notes_generator.models.util import MyDataParallel
from notes_generator.training import train
from notes_generator.training.mlflow import MlflowRunner
from notes_generator.training.loader import OnsetLoader, \
    get_difficulty_type_enum

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(lineno)d:%(message)s"
)


def load_pretrain_model(model, state_dict_path: str, device: str):
    logger.info(f"loading pretrained model from {state_dict_path}")
    model_dict = torch.load(state_dict_path, map_location=device)
    new_dict = OrderedDict()
    excludes = ("onset_linear", "onset_sequence")
    for key in model_dict:
        is_exclude = any(key.startswith(ex) for ex in excludes)
        if not is_exclude:
            new_dict[key] = model_dict[key]
    model.load_state_dict(new_dict, strict=False)
    logger.info(f"successfully loaded pretrained model.")
    return


def parse_int_bool(param_name: str, value: int) -> bool:
    if value == 0:
        return False
    elif value == 1:
        return True
    else:
        raise ValueError(f"Parameter {param_name} must be either 0 or 1.")


def parse_args(parser=None):
    parser = argparse.ArgumentParser() if parser is None else parser
    parser.add_argument("--app_name", type=AppName, default=AppName.STEPMANIA)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--score_dir", type=str)
    parser.add_argument("--mel_dir", type=str)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr_start", type=float, default=5e-7)
    parser.add_argument("--lr_end", type=float, default=5e-6)
    parser.add_argument("--seq_length", type=int, default=1600)
    parser.add_argument("--aug_count", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--onset_weight", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fuzzy_width", type=int, default=1)
    parser.add_argument("--fuzzy_scale", type=float, default=1.0)
    parser.add_argument("--with_beats", type=int, default=1)
    parser.add_argument("--difficulties", type=str, default="")
    parser.add_argument("--send_model", type=float, default=0)
    parser.add_argument("--n_saved_model", type=int, default=40)
    parser.add_argument("--augmentation_setting", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--is_parallel", type=int, default=0)
    parser.add_argument(
        "--lr_scheduler", type=str, default="CyclicLR", choices=("CyclicLR", "CosineAnnealingLR")
    )
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument(
        "--conv_stack_type", type=ConvStackType, default=ConvStackType.v1, choices=ConvStackType
    )
    parser.add_argument("--rnn_dropout", type=float, default=0)
    parser.add_argument("--pretrained_model_path", default=None)
    parser.add_argument("--disable_mlflow", action="store_true")
    args = parser.parse_args()
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    root = Path(Path(__file__).parent / "..").resolve()
    model_dir_default = {
        AppName.STEPMANIA_F: root / "data/step_mania/model",
        AppName.STEPMANIA_I: root / "data/step_mania/model",
        AppName.STEPMANIA: root / "data/step_mania/model",
    }
    score_dir_default = {
        AppName.STEPMANIA_F: root / "data/step_mania/score_onsets_1",
        AppName.STEPMANIA_I: root / "data/step_mania/score_onsets_1",
        AppName.STEPMANIA: root / "data/step_mania/score_onsets_1",
    }
    mel_dir_default = {
        AppName.STEPMANIA_F: root / "data/step_mania/mel_log",
        AppName.STEPMANIA_I: root / "data/step_mania/mel_log",
        AppName.STEPMANIA: root / "data/step_mania/mel_log",
    }

    app_name = args.app_name
    model_dir = model_dir_default[app_name] if args.model_dir is None else Path(args.model_dir)
    score_dir = score_dir_default[app_name] if args.score_dir is None else Path(args.score_dir)
    mel_dir = mel_dir_default[app_name] if args.mel_dir is None else Path(args.mel_dir)
    diff_type = get_difficulty_type_enum(app_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch
    if args.n_saved_model < 1:
        raise ValueError("n_saved_model should be greater than 0")
    send_model = parse_int_bool("send_model", args.send_model)
    with_beats = parse_int_bool("with_beats", args.with_beats)
    if args.difficulties != "":
        difficulties = tuple(d.value for d in diff_type)
    else:
        from ast import literal_eval
        difficulties = literal_eval(args.difficulties)
    if args.augmentation_setting:
        mlflow.log_artifact(args.augmentation_setting)
    onset_loader = OnsetLoader(
        score_base_path=score_dir,
        audio_base_path=mel_dir,
        seq_length=args.seq_length,
        skip_step=1000,
        aug_count=args.aug_count,
        diff_types=difficulties,
        with_other_condition=False,
        with_beats=with_beats,
        distinguish_downbeats=distinguish_downbeats,
        app_name=app_name,
        augmentation_setting=args.augmentation_setting,
    )
    model = SimpleOnsets(
        input_features=NMELS,
        output_features=1,
        num_layers=args.num_layers,
        enable_condition=True,
        enable_beats=with_beats,
        dropout=args.dropout,
        onset_weight=args.onset_weight,
        inference_chunk_length=args.seq_length // FRAME,
        conv_stack_type=args.conv_stack_type,
        rnn_dropout=args.rnn_dropout,
    ).to(device)
    if args.pretrained_model_path:
        load_pretrain_model(model, args.pretrained_model_path, device)
    if args.is_parallel == 1:
        model = MyDataParallel(model)
    starttime = datetime.now()
    logger.info(f"train start: {device} {starttime}")
    train_dataset = onset_loader.dataset("train", is_shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)
    valid_dataset = onset_loader.dataset("validation", is_shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr_start, weight_decay=args.weight_decay)
    if args.lr_scheduler == "CyclicLR":
        lr_scheduler = CyclicLR(optimizer, args.lr_start, args.lr_end, 1000, cycle_momentum=False)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * 100, eta_min=args.eta_min)
    else:
        raise ValueError
    writer = SummaryWriter(model_dir)
    model_dir = Path(model_dir)
    for f in model_dir.glob("events.*"):
        f.unlink()
    train.train_ignite(
        args.epochs,
        model,
        model_dir,
        batch_size,
        train_loader,
        valid_loader,
        optimizer,
        lr_scheduler,
        writer,
        device,
        onset_only=True,
        fuzzy_width=args.fuzzy_width,  # if >1 fuzzy label is enabled
        fuzzy_scale=args.fuzzy_scale,
        resume_checkpoint=args.resume,
        lr_find=False,
        warmup_steps=args.warmup_steps,
        send_model=send_model,
        n_saved_model=args.n_saved_model,
        eval_tolerance=eval_tolerance,
        disable_mlflow=args.disable_mlflow,
    )
    endtime = datetime.now()
    logger.info(f"train complete: {endtime - starttime}")


if __name__ == "__main__":
    with MlflowRunner(parse_args) as mr:
        main(mr.args)
