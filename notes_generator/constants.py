import enum
from typing import List, NamedTuple, Optional

##################
# common settings
##################

FRAME = 32
SAMPLE_RATE = 16000
HOP_LENGTH = 512
NMELS = 229
NOTES_COUNT = 12
MAX_THRESHOLD = 0.7


class AppName(enum.Enum):
    STEPMANIA_F = "STEPMANIA_F"
    STEPMANIA_I = "STEPMANIA_I"
    STEPMANIA = "STEPMANIA"


class ConvStackType(enum.Enum):
    v1 = "v1"
    v7 = "v7"


# Whether distinguish downbeats and other beats in beat array
distinguish_downbeats = False

# Maximum error in milliseconds in onset timing allowed during evaluation
eval_tolerance = 0.02


# loader configs
class DataIds(NamedTuple):
    train_ids: Optional[List[int]]
    validation_ids: List[int]
    test_ids: List[int]


def get_default_dataids(all_live_ids: List[int], app_name: AppName) -> DataIds:
    validation_ids = default_validation_ids[app_name]
    test_ids = default_test_ids[app_name]
    train_ids = sorted(list((set(all_live_ids) - set(test_ids + validation_ids))))

    # 10000 ~ 19999: Fraxtil's Arrow Arrangements (STEPMANIA_F)
    # 20000 ~ 29999: Fraxtil's Beast Beats        (STEPMANIA_F)
    # 30000 ~ 39999: Tsunamix III                 (STEPMANIA_F)
    # 40000 ~ 49999: In The Groove                (STEPMANIA_I)
    # 50000 ~ 59999: In The Groove 2              (STEPMANIA_I)
    if app_name == AppName.STEPMANIA_F:
        train_ids = [live_id for live_id in train_ids if 10000 <= live_id <= 39999]
    elif app_name == AppName.STEPMANIA_I:
        train_ids = [live_id for live_id in train_ids if 40000 <= live_id <= 59999]
    return DataIds(train_ids, validation_ids, test_ids)


default_test_ids = {
    AppName.STEPMANIA_F: [10011, 10019, 20009, 20016, 30019, 30032, 30036, 30038, 30042],
    AppName.STEPMANIA_I: [
        40002,
        40022,
        40037,
        40059,
        40062,
        40065,
        50017,
        50027,
        50031,
        50045,
        50048,
        50055,
    ],
    AppName.STEPMANIA: [
        10011,
        10019,
        20009,
        20016,
        30019,
        30032,
        30036,
        30038,
        30042,
        40002,
        40022,
        40037,
        40059,
        40062,
        40065,
        50017,
        50027,
        50031,
        50045,
        50048,
        50055,
    ],
}

default_validation_ids = {
    AppName.STEPMANIA_F: [10010, 20004, 20013, 30031, 30034, 30035, 30039, 30041, 30048],
    AppName.STEPMANIA_I: [
        40011,
        40032,
        40050,
        40054,
        40058,
        40061,
        50004,
        50028,
        50035,
        50049,
        50054,
        50056,
    ],
    AppName.STEPMANIA: [
        10010,
        20004,
        20013,
        30031,
        30034,
        30035,
        30039,
        30041,
        30048,
        40011,
        40032,
        40050,
        40054,
        40058,
        40061,
        50004,
        50028,
        50035,
        50049,
        50054,
        50056,
    ],
}


#####################
# StepMania settings
#####################

sm_min_note_distance = {  # frames
    10: 5,
    20: 4,
    30: 4,
    40: 4,
    50: 4,
}


sm_init_threshold = {
    10: 0.25,
    20: 0.25,
    30: 0.4,
    40: 0.4,
    50: 0.4,
}


sm_max_notes = {
    10: 80,
    20: 120,
    30: 180,
    40: 180,
    50: 180,
}


STEPMANIA_MIDI_RESOLUTION = 960


class SMDifficultyType(enum.Enum):
    beginner = 10
    easy = 20
    medium = 30
    hard = 40
    challenge = 50


class SMNotesType(enum.Enum):
    normal = 1
    hold_roll_head = 2
    hold_roll_tail = 3


def get_difficulty_type_enum(app_name: AppName):
    return SMDifficultyType
