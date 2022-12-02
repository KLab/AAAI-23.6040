import argparse
import os
import sys
import traceback
import typing
from pathlib import Path

import mlflow

ArgParserFunc = typing.Callable[[typing.Optional[argparse.ArgumentParser]], argparse.Namespace]


class MlflowRunner:
    def __init__(self, fn_args: ArgParserFunc):
        self.fn_args = fn_args
        self.args = None

    def __enter__(self):
        try:
            run_name = os.getenv("MLFLOW_RUN_NAME")
            mlflow.start_run(run_name=run_name)
            if run_name:
                mlflow.set_tag("mlflow.runName", run_name)
            # record the commit hash of data repository
            data_version = os.getenv("DATA_VERSION")
            mlflow.set_tag("data.version", data_version)
            mlflow.log_artifact("MLproject")

            parser = argparse.ArgumentParser()
            parser.add_argument("--log_artifacts", type=str)
            self.args = self.fn_args(parser)
            return self
        except:
            self.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        # executed when an error occurred
        if exc_type is not None:
            # Send error logs to Mlflow server
            traceback.print_exc(file=open("errlog.txt", "w"))
            mlflow.log_artifact("errlog.txt")

        sys.stdout.flush()
        sys.stderr.flush()
        if self.args.log_artifacts:
            # Since `mlflow run` command cannot parse multiple arguments
            # for single option, pass comma-separated string alternatively
            # and parse the string here.
            for filepath in self.args.log_artifacts.split(","):
                print(f"filepath: {filepath}")
                if Path(filepath).exists():
                    mlflow.log_artifact(filepath)
