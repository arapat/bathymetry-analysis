from .common import logger
from .common import TRAINING_FILES_DESC
from .common import VALIDATION_FILES_DESC
from .common import TESTING_FILES_DESC


config = {
    "base_dir": "./",
    "training_files": TRAINING_FILES_DESC,
    "validation_files": VALIDATION_FILES_DESC,
    "testing_files": TESTING_FILES_DESC,
    "num_leaves": 31,
    "rounds": 1000,
    "early_stopping_rounds": 200,
    "max_bin": 255,
}

