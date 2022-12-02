from pathlib import Path

from notes_generator.constants import AppName
from notes_generator.preprocessing.onset_converter import main as convert


def main(app_name: str, data_path: str, save_path: str):
    if app_name == "stepmania":
        convert(data_path, save_path, AppName.STEPMANIA)
    elif app_name == "stepmania_i":
        convert(data_path, save_path, AppName.STEPMANIA_I)
    elif app_name == "stepmania_f":
        convert(data_path, save_path, AppName.STEPMANIA_F)
    else:
        raise ValueError(f"app name {app_name} is not expected.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("app_name", choices=["stepmania", "stepmania_i", "stepmania_f"])
    parser.add_argument("-o", "--save_path")
    parser.add_argument("-u", "--data_path")
    args = parser.parse_args()

    root = (Path(__file__).parent / "..").resolve()
    data_path = args.data_path
    save_path = args.save_path

    main(args.app_name, data_path, save_path)
