from robustabstain.utils.transforms import DATA_AUG


# match an string of a perturbation region (e.g. 0.25, 1/255, 2_255, etc.)
EPS_STR_RE = r"(?:\d+(?:\.|_|\/)\d+)|\d+"

# match data augmentation string
DATA_AUG_RE = r"(?:" + r"|".join(DATA_AUG) + r")"

# match the directory of a trained model
MODEL_DIR_RE = r"^(?P<name>\S+?(?=__))(?P<timestamp>__\d{8}_\d{4})?$"

# match the directory of an abstain trained exported model (yes this is ugly ;))
ABSTAIN_MODEL_DIR_RE = ''.join([
    r"^(",
        rf"(?P<mode>[^\s_]+{EPS_STR_RE})", # match the abstain train mode (e.g. mra4_255)
        rf"(?:_(?P<dataaug>{DATA_AUG_RE}))?", # match the dataaugmentation
    r"__)?",
    r"(?P<name>\S+?(?=__))", # match the model name
    r"(?:__(?P<timestamp>\d{8}_\d{4}))?$" # match the timestamp
])

# get the top-level directory name in 'data' directory for a dataset
DATASET_TOPDIR_RE = r'^(?P<name>[^\s_]+)(?:_l)?$'

# match a filename of synthetic/original SBB cropped patches
GEN_PATCH_FNAME_RE = r'^(?P<source_img>\S+?)(?:_(?P<gen_id>\d+|original|orig))?_(?P<patch_id>\d+)\.\S+$'