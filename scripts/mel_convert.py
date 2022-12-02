from ast import literal_eval
from pathlib import Path

import click
import pandas as pd

from notes_generator.preprocessing import mel


@click.group()
def cmd():
    pass


root = Path(__file__).parent.parent


@cmd.command("single")
@click.option("--mel_save_dir", type=Path, default=root / "data/mel_log")
@click.option("--aug_count", type=int, default=0)
@click.option("--noise_rate", type=float, default=0.005)
@click.option("--bpm_info", type=str)
@click.argument("wav_path", type=Path)
@click.argument("live_id", type=int)
def single_main(
    mel_save_dir: Path,
    aug_count: int,
    noise_rate: float,
    bpm_info: str,
    wav_path: Path,
    live_id: int,
):
    assert wav_path.exists()
    assert wav_path.name.endswith(".wav")
    if bpm_info:
        bpm_info = literal_eval(bpm_info)
    mel.convert(
        wav_path.stem,
        live_id,
        wav_path.parent,
        mel_save_dir / str(live_id),
        noise_rate=noise_rate,
        aug_count=aug_count,
        bpm_info=bpm_info,
    )


@cmd.command("all")
@click.option("--mel_save_dir", type=Path, default=root / "data/mel_log")
@click.option("--wav_base_path", type=Path, default=root / "data/wav")
@click.option("--m_live_data_path", type=Path, default=root / "data/m_live_data.csv")
@click.option("--aug_count", type=int, default=0)
@click.option("--noise_rate", type=float, default=0.005)
@click.option("--parallel", is_flag=True, type=bool, default=False)
def all_main(
    mel_save_dir: Path,
    wav_base_path: Path,
    m_live_data_path: Path,
    aug_count: int,
    noise_rate: float,
    parallel: bool,
):
    df_live_data = pd.read_csv(m_live_data_path, dtype={"bpm": str})
    print(df_live_data)
    if parallel:
        mel.convert_all_parallel(
            m_live_data=df_live_data,
            wav_base_path=wav_base_path,
            save_base_path=mel_save_dir,
            aug_count=aug_count,
            log_enable=True,
            noise_rate=noise_rate,
        )
    else:
        mel.convert_all(
            m_live_data=df_live_data,
            wav_base_path=wav_base_path,
            save_base_path=mel_save_dir,
            aug_count=aug_count,
            log_enable=True,
            noise_rate=noise_rate,
        )


if __name__ == "__main__":
    cmd()
