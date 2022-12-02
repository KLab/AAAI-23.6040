import argparse
import os
from datetime import datetime
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from notes_generator.constants import *
from notes_generator.models.onsets import SimpleOnsets
from notes_generator.training.evaluate import evaluate_test
from notes_generator.training.loader import OnsetTestDataset
from notes_generator.training.model_tester import (
    LoaderConfig,
    ModelConfig,
    ModelTester,
    load_local_models,
)


class OnsetLoaderConfig(LoaderConfig):
    score_base_path: Path
    audio_base_path: Path
    live_ids: List[int]
    with_beats: bool
    app_name: AppName


class OnsetModelConfig(ModelConfig):
    input_features: int
    output_features: int
    num_layers: int
    enable_condition: bool
    enable_beats: bool
    inference_chunk_length: int
    onset_weight: int
    conv_stack_type: ConvStackType


class OnsetModelTester(ModelTester):
    def __init__(self):
        super(OnsetModelTester, self).__init__()

    def _evaluate(self, model, loaders, difficulties, device_name):
        return evaluate_test(model, loaders, difficulties, device_name)

    def _get_test_loader(self, loader_config: LoaderConfig, difficulty: int, batch_size: int):
        dataset = OnsetTestDataset(diff_type=difficulty, **loader_config._asdict())
        return DataLoader(dataset, batch_size, shuffle=False)

    def _get_test_model(self, model_config: ModelConfig, device: str):
        return SimpleOnsets(**model_config._asdict()).to(device)

    def _load_models(self, model_dir: str):
        # Saving pattern for local files
        local_model_dir = Path(model_dir) / "checkpoint"
        if local_model_dir.exists():
            model_list, model_source = load_local_models(local_model_dir, "model")
        else:
            raise FileNotFoundError(f"model directory doesn't exist: {local_model_dir}")
        return model_list, model_source


def main():
    # Path & parameter settings
    root = Path(Path(__file__).parent / "..").resolve()
    job_id = os.getenv("PJM_JOBID")
    if job_id is None:
        identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        identifier = job_id
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--app_name", type=AppName, default=AppName.STEPMANIA)
    parser.add_argument("--score_dir", type=str)
    parser.add_argument("--mel_dir", type=str)
    parser.add_argument("--seq_length", type=int, default=20480)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--onset_weight", type=int, default=64)
    parser.add_argument("--with_beats", type=int, default=1)
    parser.add_argument(
        "--conv_stack_type", type=ConvStackType, default=ConvStackType.v1, choices=ConvStackType
    )
    parser.add_argument("--csv_save_dir", default=str(os.getcwd()))
    parser.add_argument("--experiment_name", default=os.getenv("MLFLOW_EXPERIMENT_NAME"))
    args = parser.parse_args()

    model_dir = args.model_dir
    app_name = args.app_name
    score_dir = Path(args.score_dir)
    mel_dir = Path(args.mel_dir)
    seq_length = args.seq_length
    batch_size = args.batch
    csv_file_path = str(os.path.join(args.csv_save_dir, identifier + "-result.csv"))
    experiment_name = args.experiment_name
    num_layers = args.num_layers
    onset_weight = args.onset_weight
    with_beats = False if args.with_beats == 0 else 1
    conv_stack_type = args.conv_stack_type
    inference_chunk_length = seq_length // FRAME

    model_config = OnsetModelConfig(
        input_features=NMELS,
        output_features=1,
        num_layers=num_layers,
        enable_condition=True,
        enable_beats=with_beats,
        inference_chunk_length=inference_chunk_length,
        onset_weight=onset_weight,
        conv_stack_type=conv_stack_type,
    )
    loader_config = OnsetLoaderConfig(
        score_base_path=score_dir,
        audio_base_path=mel_dir,
        live_ids=default_test_ids[app_name],
        with_beats=with_beats,
        app_name=app_name,
    )

    evaluator = OnsetModelTester()
    evaluator.evaluate(
        model_dir=model_dir,
        model_config=model_config,
        loader_config=loader_config,
        app_name=app_name,
        batch_size=batch_size,
        csv_file_path=csv_file_path,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    """Tool for test models.

    This script runs test task for a specific run, given a directory containing steps, the script will test all models
    and save result to both mlflow tracking server and local csv file.

    Example
    -------
        $ python3 scripts/model_test.py data/onset_models

    Attributes
    ----------
    model_dir : str
        The path of the models to be tested.
        If <model_dir>/checkpoint exists, we assume its from local storage, and the 'checkpoint' sub-directory is
        automatically added.

    app_name : str
        The name of the game

    mel_dir : str
        The path of the mel-spectrogram data.

    score_dir : str
        The path of the onset data.

    seq_length : int
        The sequence for the LSTM layer

    batch : int
        The minibatch size of the data.

    num_layers : int
        The number of hidden layers for the LSTM layer.

    onset_weight : int

    with_beats : int
        Specify `1` or `0`.
        If `1`, the beat array will be included to the input of the model.

    conv_stack_type : str
        The type of ConvStack layer.

    csv_save_dir : str
        The base directory of saving path of the output csv file.

    experiment_name : str
        The experiment name on mlflow tracking server.
    """
    main()
