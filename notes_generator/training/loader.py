import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from notes_generator.constants import *
from notes_generator.models.beats import gen_beats_array
from notes_generator.training import augmenation


def load(base_dir: Path, app_name: AppName):
    difficulty_types = get_difficulty_type_enum(app_name)
    data_path = base_dir / "dump.npz"
    meta_path = base_dir / "meta.json"
    with meta_path.open() as fp:
        metadata = json.load(fp)
    with data_path.open("rb") as fp:
        data = np.load(fp)
        # validation
        # check if all keys in data are defined in enum `difficulty_types`
        for key in data:
            assert (
                key in difficulty_types.__members__
            ), f"Wrong onset data: '{key}' is not defined in enum {difficulty_types}."
        # load
        scores = {d.value: data[d.name] for d in difficulty_types if d.name in data}
    return metadata, scores


def load_audio(base_dir: Path, npz_file_pointer):
    metadata = load_audio_meta(base_dir)
    data = np.load(npz_file_pointer)
    return metadata, data


def load_audio_meta(base_dir: Path):
    meta_path = base_dir / "meta.json"
    with meta_path.open() as fp:
        metadata = json.load(fp)
    return metadata


def iter_array(array, length, skip, params):
    max_leng = len(array)
    for i in range(skip, max_leng, length):
        data = array[i : (i + length)]
        if len(data) == length:
            yield data, i, i + length, params


def convert_score(score):
    """Reshape onset data
    Parameters
    ----------
    score

    Returns
    -------

    """
    ret = dict(label=score)
    ret["onset"] = np.array((ret["label"] == 3), dtype=np.float)
    ret["offset"] = np.array((ret["label"] == 1), dtype=np.float)
    ret["frame"] = np.array((ret["label"] > 1), dtype=np.float)
    return ret


def assert_length(arr, length: int):
    """Pad with zeros to ensure the array length
    Parameters
    ----------
    arr
    length

    Returns
    -------

    """
    if arr.shape[0] >= length:
        return arr[:length]
    length2 = length - arr.shape[0]
    arr2 = np.zeros((length2, arr.shape[1]))
    return np.concatenate([arr, arr2])


def validate_diff_types(diff_types, app_name):
    diff_type_enum = get_difficulty_type_enum(app_name)
    valid_values = [diff.value for diff in diff_type_enum]
    for d in diff_types:
        assert d in valid_values, f"diff_type: {d} is not defined within {app_name}"
    return diff_types


class BaseLoader:
    """The base class for onset loader.

    Parameters
    ----------
    score_base_path : Path
        A path to a directory containing onset labels.
        Directories named live IDs are expected just below a level of the directory.
    audio_base_path : Path
        A path to a directory containing mel spectrogram data.
        Directories named live IDs are expected just below a level of the directory.
    device : Optional[str]
        A desired device name where training is run. Default to `None`
    seq_length : int
        The desired length of the sequence loaded from this loader.
        Default to `16000`.
        (note) The length must be calculated in wav data form.
        For example, if `10` frames are desired, specify `10 * FRAME` here.
    skip_step : int
        If `random_skip` is `False`, data will be loaded iteratively until the end,
        shifting the start position by this value.
        Default to `2000`.
        (note) The length must be calculated in wav data form.
        For example, if `10` frames are desired, specify `10 * FRAME` here.
        In addition, skip_step must satisfy following condition:
            `FRAME <= skip_step <= seq_length`
    test_split : bool
        If `True`, only specified live IDs to `data_ids` are loaded.
        Otherwise, all live_IDs of existing train data are loaded.
        Default to `True`.
    aug_count : int
        A desired number of augmented data to be included in loaded data.
        Default to `0`.
    diff_types : Tuple
        Desired condition values to be loaded.
        Condition values must be ones defined in `DifficultyType` like enum in constants.py.
        Default to `(30,)`.
    debug : bool
        If `True`, more output will be displayed in console for debugging purpose.
        Default to `False`
    random_skip : bool
        If `True`, at each iteration data will be loaded from random start position until the end.
        Default to `True`.
    with_other_condition : bool
        If `True`, data for all conditions are included to 'other_conditions' key
        in the return dict. Default to `False`.
    with_beats : bool
        If `True`, a beat array generated from `bpm_info` is included to 'beats' key
        in the return dict. Default to `False`.
    app_name : AppName
        A desired app name of data to be loaded. Defalt to `AppName.STEPMANIA`.
    data_ids : Optional[DataIds]
        If `test_split` is `True`, data of live IDs specified here will be loaded.
        If `None`, `default_validation_ids` and `default_test_ids` in constants.py are set to
        validation IDs and test IDs respectively, and the other songs are set to train IDs.
        Default to `None`.
    augmentation_setting: Optional[str]
        setting file.
    """

    def __init__(
        self,
        score_base_path: Path,
        audio_base_path: Path,
        device: str = None,
        seq_length: int = 16000,
        skip_step: int = 2000,
        test_split: bool = True,
        aug_count: int = 0,
        diff_types: Tuple = (30,),
        debug: bool = False,
        random_skip: bool = True,
        with_other_condition: bool = False,
        with_beats: bool = False,
        distinguish_downbeats: bool = True,
        app_name: AppName = AppName.STEPMANIA,
        data_ids: Optional[DataIds] = None,
        augmentation_setting: Optional[str] = None,
    ):
        self.score_base_path = Path(score_base_path)
        self.audio_base_path = Path(audio_base_path)
        self.seq_length = seq_length
        self.skip_step = skip_step
        if device is not None:
            warnings.warn("Loader device parameter is obsolete. It should be None")
            device = None
        self.device = device
        self.aug_count = aug_count
        self.diff_types = validate_diff_types(diff_types, app_name)
        self.debug = debug
        self.random_skip = random_skip
        self.with_other_condition = with_other_condition
        self.with_beats = with_beats
        self.distinguish_downbeats = distinguish_downbeats
        self.app_name = app_name
        if augmentation_setting:
            self.augmentations = augmenation.load_augmentations(Path(augmentation_setting))
        else:
            self.augmentations = None
        self._score_dict = self.load_dir(self.score_base_path)
        live_ids = list(self._score_dict.keys())
        self.live_ids = live_ids
        if test_split:
            if data_ids is None:
                data_ids = get_default_dataids(live_ids, app_name)
            assert set(data_ids.train_ids) & set(data_ids.validation_ids) == set()
            assert set(data_ids.train_ids) & set(data_ids.test_ids) == set()
            assert set(data_ids.test_ids) & set(data_ids.validation_ids) == set()
            self.train_ids = data_ids.train_ids
            self.test_ids = data_ids.test_ids
            self.validation_ids = data_ids.validation_ids
        else:
            self.train_ids = sorted(live_ids)
            self.test_ids = []
            self.validation_ids = []

    def load_dir(self, base_path: Path):
        """Load onset data for all live IDs.

        Parameters
        ----------
        base_path : Path
            A path to a directory containing onset labels.
            Directories named live IDs are expected just below a level of the directory.

        Returns
        -------
        score_dict : Dict[int, Tuple[Dict, Dict]]

        """
        score_dict = dict()
        for e in base_path.iterdir():
            if not e.is_dir():
                continue
            metadata, scores = load(e, self.app_name)

            live_id = int(metadata["live_id"])
            score_dict[live_id] = metadata, scores
        return score_dict

    def dataset(self, mode: str = "all", is_shuffle: bool = False, shuffle_buffer_size: int = 500):
        """Return Dataset class in the specified mode.

        Parameters
        ----------
        mode : str
            A mode of the loader, which should be one of 'train', 'validation' of 'test'.
            The data of live IDs specified to `data_ids` in a loader of corresponding mode
            will be loaded.
        is_shuffle : bool
            If `True`, this method will return a loader which loads each data in shuffled manner.
        shuffle_buffer_size : int
            If `is_shuffle` is `True`, a returned loader will once load as many data as
            the number of buffer size in normal order, and then shuffle the data and return.

        Returns
        -------
        Type[IterableDataset]

        """
        if is_shuffle:
            return ShuffleDataset(self, shuffle_buffer_size, mode)
        return ScoreDataset(self, mode)

    def iter(self, train_or_test: str):
        """Iterate all data.

        Parameters
        ----------
        train_or_test : str
            The set name of live IDs loaded in this loader.
            The value must be one of 'train', 'validation' or 'test.

        Returns
        -------

        """
        if train_or_test == "train":
            ids = self.train_ids
        elif train_or_test == "validation":
            ids = self.validation_ids
        elif train_or_test == "test":
            ids = self.test_ids
        else:
            raise ValueError()
        for id_ in ids:
            yield from self.iter_live(id_, train_or_test)

    def iter_live(self, live_id, train_or_test: str):
        """Iterate songs

        Parameters
        ----------
        live_id : int
            A live ID of data to be loaded.
        train_or_test : str
            The set name of live IDs loaded in this loader.
            The value must be one of 'train', 'validation' or 'test.

        Returns
        -------

        """
        raise NotImplementedError

    def iter_audio(self, live_id, train_or_test: str):
        """Iterate audio

        Parameters
        ----------
        live_id : int
            A live ID of data to be loaded.
        train_or_test : str
            The set name of live IDs loaded in this loader.
            The value must be one of 'train', 'validation' or 'test.

        Returns
        -------

        """
        path = self.audio_base_path / str(live_id)
        npz_path = path / "mel.npz"
        with npz_path.open("rb") as fpt:
            metadata, mel_data = load_audio(path, fpt)
            keys = ["mel"]
            if self.aug_count > 0 and train_or_test == "train":
                keys += [f"mel_noise_{idx}" for idx in range(self.aug_count)]
            for key in keys:
                seq_length = int(round(self.seq_length / FRAME))
                song_length = len(mel_data[key])
                if self.random_skip and train_or_test == "train":
                    skip = np.random.randint(0, seq_length)
                    params = dict(key=key, skip=skip, seq_length=seq_length, live_id=live_id)
                    yield from iter_array(mel_data[key], seq_length, skip, params)
                else:
                    skip_step = int(round(self.skip_step / FRAME))
                    for skip in range(0, seq_length, skip_step):
                        params = dict(key=key, skip=skip, seq_length=seq_length, live_id=live_id)
                        yield from iter_array(mel_data[key], seq_length, skip, params)


class OnsetLoader(BaseLoader):
    def cut_segment(self, score_data: np.ndarray, start_index: int, end_index: int, length: int):
        """Ensure the length of data array.

        Parameters
        ----------
        score_data : np.ndarray
            An array containing score data where the length will be adjusted.
        start_index : int
            The start index to cut the array.
        end_index : int
            The end index to cut the array.
        length : length
            The length of the array to be returned.

        Returns
        -------

        """
        # Cut the score data by specified length
        score_segment = score_data[start_index:end_index]
        # Pad if the length is insufficient
        score_segment = assert_length(score_segment, length)
        return score_segment

    def handle_other_conditions(
        self, score_dict: Dict, start_index: int, end_index: int, length: int
    ):
        """Return a dict containing data of all conditions.

        Parameters
        ----------
        score_dict : Dict
            A dict whose key is expected to be condition value,
            and whose value is expected to be an array containing score data.
        start_index : int
            The start index to cut the arrays in `score_dict`.
        end_index : int
            The end index to cut the arrays in `score_dict`.
        length : int
            The length of arrays for each conditions to be returned.

        Returns
        -------
        Dict

        """
        # Add data of other difficulties for a convenience
        dic = {
            diff_type: torch.from_numpy(
                self.cut_segment(score_dict[diff_type], start_index, end_index, length)
            ).float()
            for diff_type in self.diff_types
            if diff_type in score_dict.keys()
        }
        return dic

    def get_bpm_info(self, live_id):
        """Return bpm information for specified live ID.

        Parameters
        ----------
        live_id : int
            A live ID whose bpm information will be returned.

        Returns
        -------

        """
        path = self.audio_base_path / str(live_id)
        metadata = load_audio_meta(path)
        return metadata["bpm_info"]

    def iter_live(self, live_id, train_or_test: str):
        """Iterate songs
        Parameters
        ----------
        live_id : int
            A live ID of data to be loaded.
        train_or_test : str
            The set name of live IDs loaded in this loader.
            The value must be one of 'train', 'validation' or 'test.

        Returns
        -------

        """
        # score_dic = {10: array([[0.], ...]), 20: array([[0.], ...]), ...}
        _, score_dic = self._score_dict[live_id]
        bpm_info = self.get_bpm_info(live_id)
        onsets_array_len = len(list(score_dic.values())[0])
        audio_path = self.audio_base_path / str(live_id)
        audio_meta = load_audio_meta(audio_path)
        mel_len = round(audio_meta["mel_length"] * 1000)  # ms
        beats_array = gen_beats_array(
            onsets_array_len, bpm_info, mel_len, self.distinguish_downbeats
        )
        for diff_type in self.diff_types:
            condition = diff_type
            if diff_type in score_dic:
                score_data = score_dic.get(diff_type)
            else:
                continue

            for audio, start_index, end_index, params in self.iter_audio(live_id, train_or_test):
                score_segment = self.cut_segment(score_data, start_index, end_index, audio.shape[0])
                # Convert to the form of start, end, frame
                data = dict(
                    condition=torch.Tensor([condition]),  # difficulty
                    onset=torch.from_numpy(score_segment).float(),
                    audio=torch.from_numpy(audio).float(),  # audio (mel-spectrogram)
                )
                if self.with_other_condition:
                    data["other_conditions"] = self.handle_other_conditions(
                        score_dic, start_index, end_index, audio.shape[0]
                    )
                if self.with_beats:
                    # beat array(2 at downbeats, 1 at other beats)
                    data["beats"] = torch.from_numpy(
                        self.cut_segment(beats_array, start_index, end_index, audio.shape[0])
                    ).float()
                if self.augmentations and train_or_test == "train":
                    data = augmenation.apply_augmentation(self.augmentations, data)
                if self.debug:
                    data.update(params)
                    data["start_index"] = start_index
                    data["end_index"] = end_index
                yield data


class ScoreDataset(torch.utils.data.IterableDataset):
    def __init__(self, score_loader: BaseLoader, mode: str):
        self.score_loader = score_loader
        self.mode = mode

    def __iter__(self):
        return iter(self.score_loader.iter(self.mode))


# https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130
class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, score_loader: BaseLoader, buffer_size: int, mode: str):
        self.score_loader = score_loader
        self.buffer_size = buffer_size
        self.mode = mode

    def __iter__(self):
        shufbuf = []
        dataset_iter = iter(self.score_loader.iter(self.mode))
        try:
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


class OnsetTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        score_base_path: Path,
        audio_base_path: Path,
        live_ids: List[int],
        device: str = None,
        diff_type: int = 30,
        with_beats: bool = True,
        app_name: AppName = AppName.STEPMANIA,
    ):
        super().__init__()
        self.score_base_path = score_base_path
        self.audio_base_path = audio_base_path
        self.live_ids = live_ids
        self.device = device
        self.diff_type = diff_type
        self.with_beats = with_beats
        self.app_name = app_name
        self.score_dict = self.load_score_dict()
        self.exist_live_ids = list(self.score_dict.keys())

    def load_score_dict(self):
        score_dict = dict()
        for live_id in self.live_ids:
            try:
                score = self.load_score(live_id)
                score_dict[live_id] = score
            except KeyError:
                continue

        return score_dict

    def load_audio(self, live_id):
        audio_dir = self.audio_base_path / str(live_id)
        npz_path = audio_dir / "mel.npz"
        with npz_path.open("rb") as fpt:
            metadata, mel_data = load_audio(audio_dir, fpt)
            audio = mel_data["mel"]
        return audio, metadata

    def load_score(self, live_id):
        score_dir = self.score_base_path / str(live_id)
        metadata, scores = load(score_dir, self.app_name)
        return scores[self.diff_type]

    def __getitem__(self, index):
        live_id = self.exist_live_ids[index]

        # load audio
        audio, audio_meta = self.load_audio(live_id)
        audio_len = len(audio)
        bpm_info = audio_meta["bpm_info"]

        # load score
        score = self.score_dict[live_id]
        score = assert_length(score, audio_len)

        data = {
            "live_id": live_id,
            "audio": torch.from_numpy(audio).float(),
            "onset": torch.from_numpy(score).float(),
            "condition": torch.tensor([self.diff_type]),
        }
        if self.with_beats:
            beats_array = gen_beats_array(audio_len, bpm_info, audio_len)
            data["beats"] = torch.from_numpy(beats_array).float()
        return data

    def __len__(self):
        return len(self.exist_live_ids)
