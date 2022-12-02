import csv
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, TextIO, Type, Union

import mlflow
import torch
import torch.multiprocessing as mp
import yaml
from torch import nn
from torch.utils.data.dataloader import DataLoader

from notes_generator.constants import *

LoaderConfig = NamedTuple
ModelConfig = NamedTuple
EvalDict = Dict[str, Dict[str, Union[int, float]]]  # {diff_name: {metric_name: metric_value}}


class ModelInfo(NamedTuple):
    model_name: str
    model_path: str
    step: int


def load_local_models(model_dir: Path, prefix: str):
    model_list = []
    glob_ptn = prefix + "*"
    for model_path in model_dir.glob(glob_ptn):
        model_list.append(
            ModelInfo(
                model_name=model_path.stem,
                model_path=str(model_path),
                step=int(model_path.stem.split("_")[-1]),
            )
        )
    return model_list, "local"


def load_mlflow_models(model_dir: str):
    model_list = []
    # Traditional way to search for models in directories
    for relative_path, sub_dir, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".pth"):
                ml_model_path = Path(relative_path).parent / "MLmodel"
                with open(ml_model_path, "r") as model_info_file:
                    model_info = yaml.safe_load(model_info_file)
                    run_id = model_info["run_id"]
                    artifact_path = model_info["artifact_path"]
                    if artifact_path.split("_")[-1].isnumeric():
                        step = int(artifact_path.split("_")[-1])
                    else:
                        step = 0
                model_list.append(
                    ModelInfo(
                        model_name=run_id + "_" + artifact_path,
                        model_path=os.path.join(relative_path, file),
                        step=step,
                    )
                )
    return model_list, "mlflow"


def _save_result(csv_file: TextIO, model_source: str, result_dict: Dict):
    header_written = False
    csv_writer = csv.writer(csv_file)
    if model_source == "local":
        header = ["model_name", "difficulty"]
        for key in result_dict:
            model_name = result_dict[key]["model_name"]
            for difficulty_key in result_dict[key]["eval_result"].keys():
                row = [model_name, difficulty_key]
                if not header_written:
                    header.extend(result_dict[key]["eval_result"][difficulty_key].keys())
                    csv_writer.writerow(header)
                    header_written = True
                row.extend(result_dict[key]["eval_result"][difficulty_key].values())
                csv_writer.writerow(row)
                # Log metrics one-by-one to mlflow tracking server
                for metric_key in result_dict[key]["eval_result"][difficulty_key]:
                    metric_name = difficulty_key + "/" + metric_key
                    metric_scalar = result_dict[key]["eval_result"][difficulty_key][metric_key]
                    print(f"Writing metric: {metric_name}@{model_name}")
                    mlflow.log_metric(
                        key=metric_name,
                        value=metric_scalar,
                        step=result_dict[key]["step"],
                    )
    elif model_source == "mlflow":
        header = ["run_id", "difficulty"]
        for key in result_dict:
            run_id = result_dict[key]["model_name"].split("_")[0]
            for difficulty_key in result_dict[key]["eval_result"].keys():
                row = [run_id, difficulty_key]
                if not header_written:
                    header.extend(result_dict[key]["eval_result"][difficulty_key].keys())
                    csv_writer.writerow(header)
                    header_written = True
                row.extend(result_dict[key]["eval_result"][difficulty_key].values())
                csv_writer.writerow(row)
                # Log metrics one-by-one to mlflow tracking server
                for metric_key in result_dict[key]["eval_result"][difficulty_key]:
                    metric_name = run_id + "/" + difficulty_key + "/" + metric_key
                    metric_scalar = result_dict[key]["eval_result"][difficulty_key][metric_key]
                    print(f"Writing metric: {metric_name}@{key}")
                    mlflow.log_metric(
                        key=metric_name,
                        value=metric_scalar,
                        step=result_dict[key]["step"],
                    )
    else:
        raise NotImplementedError("Invalid model source")


class ModelTester:
    """This class is for performing batch evaluation.

    Usage
    -----
    1. Make a subclass inheriting this class
    2. Override methods below and implement concrete method:
        * _evaluate
        * _get_test_loader
        * _get_test_models
        * _load_models
    3. Run evaluate()
    """

    def _evaluate(
        self,
        model: nn.Module,
        loaders: List[DataLoader],
        difficulties: Type[Enum],
        device_name: str,
    ) -> EvalDict:
        """This method implements concrete evaluation process.

        Parameters
        ----------
        model : nn.Module
            The model which is evaluated.
        loaders : List[DataLoader]
            The loader class feeding test data.
        difficulties : Type[Enum]
            The Enum class defining difficulty list for each game.
        device_name : str
            Device used for evaluation, should be acceptable by torch.device()

        Returns
        -------
        result_dict : EvalDict
            The dictionary containing evaluation results.
            The structure is below:
                {diff_name: {metric_key: metric_value}}

        """
        raise NotImplementedError

    def _get_test_loader(
        self, loader_config: LoaderConfig, difficulty: int, batch_size: int
    ) -> DataLoader:
        """This method implements how to load a data loader class.

        Parameters
        ----------
        loader_config : LoaderConfig
            The named tuple containing loader configurations
        difficulty : int
            The difficulty level as integer mapping to constant.py
        batch_size : int
            The size of minibatch used in evaluation.

        Returns
        -------
        data_loader : DataLoader
            The data loader class used in evaluation
        """
        raise NotImplementedError

    def _get_test_model(self, model_config: ModelConfig, device: str):
        """This method implements how to load a concrete model with parameters.

        Parameters
        ----------
        model_config : ModelConfig
            The named tuple containing model configurations
        device : str
            Device used for evaluation, should be acceptable by torch.device()

        Returns
        -------
        model: nn.Module
            The model object for the specified model path.
            The model is assumed that either trained parameters have been loaded
            and the model has been sent to specified device.
        """
        raise NotImplementedError

    def _load_models(self, model_dir: str):
        """This method implements how to load models in given directory.

        Parameters
        ----------
        model_dir : str
            The directory to the models to be tested

        Returns
        ----------
        Model information : List[ModelInfo]
            A list of ModelInfo containing model's information
        Model source : str
            The source of the model is automatically inferred from the given directory, if <model_dir>/checkpoint exists, we
            assume its from local saved models; otherwise we assume its from mlflow server and search for files that ends
            with ".pth" and its corresponding "MLmodel" file to get the run_id
        """
        raise NotImplementedError

    def evaluate(
        self,
        model_dir: str,
        model_config: ModelConfig,
        loader_config: LoaderConfig,
        batch_size: int,
        app_name: AppName,
        csv_file_path: str,
        experiment_name: str,
    ):
        """This method runs evaluation and save results

        Parameters
        ----------
        model_dir : str
            The directory to the models to be tested
        model_config : ModelConfig
            The named tuple containing model configurations
        loader_config : LoaderConfig
            The named tuple containing loader configurations
        batch_size : int
            The minibatch size of test data
        app_name : AppName
            The name of the game
        csv_file_path : str
            The path to csv file in which the result will be stored
        experiment_name : str
            The result will be sent to mlflow server with this experiment name
        """
        model_list, model_source = self._load_models(model_dir)

        difficulties = get_difficulty_type_enum(app_name)
        loaders = {
            d.value: self._get_test_loader(loader_config, d.value, batch_size) for d in difficulties
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            eval_result = self._evaluate_on_gpu(
                model_list,
                model_config,
                loaders,
                app_name,
            )
        else:
            eval_result = self._evaluate_on_cpu(
                model_list,
                model_config,
                loaders,
                app_name,
            )
        self._save_result(
            result_dict=eval_result,
            csv_file_path=csv_file_path,
            experiment_name=experiment_name,
            model_source=model_source,
        )

    def _evaluate_model(
        self,
        model_info: ModelInfo,
        model_config: ModelConfig,
        loaders: Dict[int, DataLoader],
        result_dictionary: Any,
        device_name: str,
        app_name: AppName,
    ):
        """This method runs evaluation on designated device.

        Parameters
        ----------
        model_info : ModelInfo
            The information of the model to be tested. Should contain name, path and whether with steps
        model_config : ModelConfig
            The named tuple containing model configurations
        loader_config : LoaderConfig
            The named tuple containing loader configurations
        result_dictionary : Dict or torch.multiprocessing.Manager.dict()
            The dictionary to save results
        device_name : str
            Device to be used for the test
        app_name : AppName
            The name of the game
        """
        print(f"Evaluating model: {model_info.model_name} on device: {device_name}")
        eval_dict = {}
        model = self._get_test_model(model_config, device_name)
        # If trained with data-parallel, should manually call state_dict
        try:
            state_dict = torch.load(model_info.model_path, map_location=torch.device(device_name))
            model.load_state_dict(state_dict)
        except:
            model.load_state_dict(
                torch.load(
                    model_info.model_path, map_location=torch.device(device_name)
                ).module.state_dict()
            )
        difficulties = get_difficulty_type_enum(app_name)
        eval_dict = self._evaluate(model, loaders, difficulties, device_name)
        result_dictionary[model_info.model_path] = {
            "eval_result": eval_dict,
            "model_name": model_info.model_name,
            "step": model_info.step,
        }

    def _evaluate_on_cpu(
        self,
        model_list: list,
        model_config: ModelConfig,
        loaders: Dict[int, DataLoader],
        app_name: AppName,
    ) -> Dict:
        """This method runs evaluation task on CPU environment.

        Parameters
        ----------
        model_list : list
            The list of the models to be tested, list of ModelInfo
        model_config : ModelConfig
            The named tuple containing model configurations
        loader_config : LoaderConfig
            The named tuple containing loader configurations
        app_name : AppName
            The name of the game

        Returns
        -------
        Dict
            Cascaded result dictionary, format:
            {model_name:{difficulty:{metric_name:value}}}
        """
        res_dict = {}
        for model_info in model_list:
            self._evaluate_model(
                model_info=model_info,
                model_config=model_config,
                loaders=loaders,
                result_dictionary=res_dict,
                device_name="cpu",
                app_name=app_name,
            )
        return res_dict

    def _evaluate_on_gpu(
        self,
        model_list: list,
        model_config: ModelConfig,
        loaders: Dict[int, DataLoader],
        app_name: AppName,
    ) -> Dict:
        """This method runs evaluation task on GPU environment.

        Parameters
        ----------
        model_list : list
            The list of the models to be tested, list of ModelInfo
        model_config : ModelConfig
            The named tuple containing model configurations
        loader_config : LoaderConfig
            The named tuple containing loader configurations
        app_name : AppName
            The name of the game

        Returns
        -------
        Dict
            Cascaded result dictionary, format:
            {model_name:{difficulty:{metric_name:value}}}
        """
        # Pytorch multiprocess settings
        mp.set_start_method("spawn")
        manager = mp.Manager()
        res_dict = manager.dict()
        gpu_count = torch.cuda.device_count()
        print(f"Running Parallel Task With Parallel Process: {gpu_count}")
        batch_count = len(model_list) // gpu_count + 1
        for batch_index in range(batch_count):
            process_list = []
            print(f"Processing Batch {batch_index + 1}, total: {batch_count}, GPUs: {gpu_count}")
            if batch_index < batch_count - 1:
                process_num = gpu_count
            else:
                process_num = len(model_list) % gpu_count
            for process_index in range(process_num):
                model_info = model_list[batch_index * gpu_count + process_index]
                assigned_device = f"cuda:{process_index}"
                print(
                    f"Starting process #{process_index}, Assigned device: {assigned_device}, model:{model_info}"
                )
                p = mp.Process(
                    target=self._evaluate_model,
                    args=(
                        model_info,
                        model_config,
                        loaders,
                        res_dict,
                        assigned_device,
                        app_name,
                    ),
                )
                process_list.append(p)
                p.start()
            for p in process_list:
                p.join()
                p.close()
            print("Batch process complete")
        return res_dict.copy()

    def _save_result(
        self,
        result_dict: dict,
        experiment_name: str,
        csv_file_path: str,
        model_source: str,
    ):
        """This method sends result to mlflow tracking server and save a copy to a csv file.

        Parameters
        ----------
        result_dict : Dict
            Result data generated by the evaluation function.
        experiment_name : str
            The path to the onset data.
        csv_file_path : str
            The csv file to save the result.
        model_source : str
            The source of the model, can be locally saved model or saved automatically by mlflow.
        """
        with mlflow.start_run():
            run_name = os.getenv("MLFLOW_RUN_NAME")
            if run_name:
                mlflow.set_tag("mlflow.runName", run_name)
            data_version = os.getenv("DATA_VERSION")
            if data_version:
                mlflow.set_tag("data.version", data_version)

        with open(csv_file_path, "w") as csv_file:
            _save_result(csv_file, model_source, result_dict)
