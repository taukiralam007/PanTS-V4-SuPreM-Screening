# ============================================================
# PanTS V4 — SuPreM PRETRAINED SegResNet
# Taukir Alam
#
# Google Colab Pro T4 unattended/resumable pipeline
#
# Improvements over V3:
# 1. SuPreM pretrained abdominal CT SegResNet
# 2. 1.5 mm isotropic spacing
# 3. 96 x 96 x 96 patches
# 4. Multi-task:
#       channel 0 = pancreas region
#       channel 1 = pancreatic lesion
# 5. Pancreas supervision ignored for cases where annotation
#    is genuinely missing
# 6. Lesion-aware Tversky + focal loss
# 7. Anatomy gating at inference
# 8. Validation-only threshold tuning
# 9. Every epoch checkpointed
# 10. Resume support
# ============================================================

import os
import sys
import glob
import json
import time
import random
import shutil
import subprocess
import importlib.util
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42

MAX_EPOCHS = 12

PATCH = (96, 96, 96)

SPACING = (1.5, 1.5, 1.5)

CT_MIN = -175
CT_MAX = 250

POSITIVE_PATIENT_WEIGHT = 3.0

# background : pancreas : lesion
CROP_RATIOS = [1, 3, 4]

GRAD_ACCUM = 2

BASE_LR = 3e-5
HEAD_LR = 2e-4

PROJECT = "/content/drive/MyDrive/PanTS_Project/v4_suprem"

TMP = "/content/PanTS/v4_work"

CHECKPOINT_DIR = f"{PROJECT}/checkpoints"
RESULTS_DIR = f"{PROJECT}/results"
MANIFEST_DIR = f"{PROJECT}/manifests"
SUBMISSION_DIR = f"{PROJECT}/submission"

for d in [
    PROJECT,
    TMP,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    MANIFEST_DIR,
    SUBMISSION_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ------------------------------------------------------------
# PanTS data
# ------------------------------------------------------------

IMAGE_ROOT = "/content/PanTS/data/ImageTr"

IMAGE_ARCHIVE = (
    "/content/PanTS/data/PanTSMini_ImageTr_00000001_00001000.tar.gz"
)

IMAGE_URL = (
    "https://huggingface.co/datasets/"
    "BodyMaps/PanTSMini/resolve/main/"
    "PanTSMini_ImageTr_00000001_00001000.tar.gz"
    "?download=true"
)

LABEL_ARCHIVE = (
    "/content/PanTS/data/PanTSMini_Label.tar.gz"
)

LABEL_URL = (
    "https://www.cs.jhu.edu/~zongwei/dataset/"
    "PanTSMini_Label.tar.gz"
)

OFFICIAL_COMBINED = (
    f"{TMP}/OfficialCombinedTr"
)

LABEL3_ROOT = (
    f"{TMP}/Label3Tr"
)

# ------------------------------------------------------------
# SuPreM
# ------------------------------------------------------------

SUPREM_WEIGHTS = (
    f"{PROJECT}/"
    "supervised_suprem_segresnet_2100.pth"
)

SUPREM_URL = (
    "https://huggingface.co/MrGiovanni/"
    "SuPreM/resolve/main/"
    "supervised_suprem_segresnet_2100.pth"
)

# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

BEST_MODEL = (
    f"{CHECKPOINT_DIR}/"
    "best_v4_suprem_segresnet.pt"
)

LAST_MODEL = (
    f"{CHECKPOINT_DIR}/"
    "last_v4_suprem_segresnet.pt"
)

HISTORY_FILE = (
    f"{RESULTS_DIR}/v4_history.csv"
)

# ============================================================
# HELPERS
# ============================================================

def run(cmd):
    print("\nRUNNING:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)


def download(url, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    run([
        "wget",
        "-c",
        "-O",
        path,
        url
    ])


# ============================================================
# INSTALL PACKAGES IF REQUIRED
# ============================================================

needed = [
    "monai",
    "nibabel",
    "scipy",
    "sklearn",
    "pandas",
]

missing = [
    x for x in needed
    if importlib.util.find_spec(x) is None
]

if missing:

    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "monai",
        "nibabel",
        "scipy",
        "scikit-learn",
        "pandas"
    ])


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from scipy import ndimage

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import monai

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    MapTransform,
)

from monai.data import (
    Dataset,
    DataLoader,
)

from monai.networks.nets import SegResNet

from monai.inferers import (
    sliding_window_inference,
)

from torch.utils.data import (
    WeightedRandomSampler,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

assert torch.cuda.is_available(), (
    "GPU unavailable. Enable a GPU runtime in Google Colab."
)

device = torch.device("cuda")


print("=" * 65)
print("ENVIRONMENT")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)
print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# PART 1 — RESTORE 1000 CT VOLUMES
# ============================================================

def count_ct():

    return len(
        glob.glob(
            f"{IMAGE_ROOT}/"
            "PanTS_*/ct.nii.gz"
        )
    )


if count_ct() != 1000:

    print(
        "\nTraining CT data missing."
    )

    print(
        "Downloading first 1000 "
        "PanTSMini CT volumes..."
    )

    if not os.path.exists(
        IMAGE_ARCHIVE
    ):

        download(
            IMAGE_URL,
            IMAGE_ARCHIVE
        )

    os.makedirs(
        IMAGE_ROOT,
        exist_ok=True
    )

    run([
        "tar",
        "-xzf",
        IMAGE_ARCHIVE,
        "-C",
        IMAGE_ROOT
    ])


print(
    "\nTraining CT volumes:",
    count_ct()
)

assert count_ct() == 1000


# Save disk after successful extraction

if os.path.exists(IMAGE_ARCHIVE):

    os.remove(IMAGE_ARCHIVE)

    print(
        "Deleted compressed CT archive."
    )


# ============================================================
# PART 2 — LABEL ARCHIVE
# ============================================================

if not os.path.exists(
    LABEL_ARCHIVE
):

    print(
        "\nDownloading official "
        "PanTS labels..."
    )

    download(
        LABEL_URL,
        LABEL_ARCHIVE
    )


print(
    "Label archive GB:",
    round(
        os.path.getsize(
            LABEL_ARCHIVE
        ) / 1024**3,
        2
    )
)


# ============================================================
# PART 3 — EXTRACT COMBINED LABELS
# ============================================================

case_ids = [
    f"PanTS_{i:08d}"
    for i in range(1, 1001)
]

os.makedirs(
    OFFICIAL_COMBINED,
    exist_ok=True
)


combined_count = len(
    glob.glob(
        f"{OFFICIAL_COMBINED}/"
        "PanTS_*/combined_labels.nii.gz"
    )
)


if combined_count != 1000:

    list_file = (
        f"{TMP}/"
        "combined_train_1000.txt"
    )

    with open(
        list_file,
        "w"
    ) as f:

        for case in case_ids:

            f.write(
                f"{case}/"
                "combined_labels.nii.gz\n"
            )


    print(
        "\nExtracting 1000 official "
        "combined labels..."
    )

    run([
        "tar",
        "-xzf",
        LABEL_ARCHIVE,
        "-C",
        OFFICIAL_COMBINED,
        "-T",
        list_file
    ])


combined_count = len(
    glob.glob(
        f"{OFFICIAL_COMBINED}/"
        "PanTS_*/combined_labels.nii.gz"
    )
)

print(
    "Official combined labels:",
    combined_count
)

assert combined_count == 1000


# ============================================================
# PART 4 — CREATE 3-CLASS SAMPLING LABEL
#
# 0 = background
# 1 = pancreas
# 2 = lesion
#
# PanTS:
# 17 pancreas
# 18 body
# 19 head
# 20 tail
# 28 pancreatic lesion
# ============================================================

os.makedirs(
    LABEL3_ROOT,
    exist_ok=True
)

print(
    "\nCreating V4 labels..."
)

for case in tqdm(
    case_ids,
    desc="V4 labels"
):

    src = (
        f"{OFFICIAL_COMBINED}/"
        f"{case}/combined_labels.nii.gz"
    )

    dst_dir = (
        f"{LABEL3_ROOT}/{case}"
    )

    os.makedirs(
        dst_dir,
        exist_ok=True
    )

    dst = (
        f"{dst_dir}/"
        "label_3class.nii.gz"
    )

    if os.path.exists(dst):
        continue

    nii = nib.load(src)

    arr = np.asarray(
        nii.dataobj
    )

    target = np.zeros(
        arr.shape,
        dtype=np.uint8
    )

    pancreas = np.isin(
        arr,
        [17, 18, 19, 20]
    )

    lesion = (
        arr == 28
    )

    target[pancreas] = 1
    target[lesion] = 2

    output = nib.Nifti1Image(
        target,
        affine=nii.affine,
        header=nii.header.copy()
    )

    output.set_data_dtype(
        np.uint8
    )

    nib.save(
        output,
        dst
    )


assert len(
    glob.glob(
        f"{LABEL3_ROOT}/"
        "PanTS_*/label_3class.nii.gz"
    )
) == 1000


# ============================================================
# PART 5 — AUDIT + MANIFEST
# ============================================================

rows = []

print(
    "\nAuditing 1000 labels..."
)

for case in tqdm(
    case_ids,
    desc="Label audit"
):

    image_path = (
        f"{IMAGE_ROOT}/"
        f"{case}/ct.nii.gz"
    )

    label_path = (
        f"{LABEL3_ROOT}/"
        f"{case}/label_3class.nii.gz"
    )

    mask = np.asarray(
        nib.load(label_path).dataobj
    )

    pancreas_voxels = int(
        (mask == 1).sum()
    )

    lesion_voxels = int(
        (mask == 2).sum()
    )

    rows.append({

        "case_id":
            case,

        "image":
            image_path,

        "label":
            label_path,

        "pancreas_voxels":
            pancreas_voxels,

        "lesion_voxels":
            lesion_voxels,

        "pancreas_valid":
            int(
                pancreas_voxels > 0
            ),

        "has_lesion":
            int(
                lesion_voxels > 0
            ),
    })


master = pd.DataFrame(rows)

master.to_csv(
    f"{MANIFEST_DIR}/"
    "master_v4.csv",
    index=False
)


print(
    "\nDATASET"
)

print(
    "Total:",
    len(master)
)

print(
    "Lesion positive:",
    int(
        master.has_lesion.sum()
    )
)

print(
    "Lesion negative:",
    int(
        (master.has_lesion == 0).sum()
    )
)

print(
    "Pancreas annotation missing:",
    int(
        (master.pancreas_valid == 0).sum()
    )
)


# ============================================================
# PART 6 — REPRODUCIBLE 800/100/100 SPLIT
# ============================================================

train_df, temp_df = (
    train_test_split(
        master,
        test_size=0.20,
        random_state=SEED,
        stratify=master[
            "has_lesion"
        ]
    )
)

val_df, test_df = (
    train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df[
            "has_lesion"
        ]
    )
)


train_df = (
    train_df
    .sort_values("case_id")
    .reset_index(drop=True)
)

val_df = (
    val_df
    .sort_values("case_id")
    .reset_index(drop=True)
)

test_df = (
    test_df
    .sort_values("case_id")
    .reset_index(drop=True)
)


for name, df in [

    ("TRAIN", train_df),
    ("VAL", val_df),
    ("TEST", test_df),

]:

    print(
        name,
        len(df),
        "| positive:",
        int(
            df.has_lesion.sum()
        ),
        "| negative:",
        int(
            (df.has_lesion == 0).sum()
        ),
        "| missing pancreas:",
        int(
            (df.pancreas_valid == 0).sum()
        )
    )


train_df.to_csv(
    f"{MANIFEST_DIR}/train_v4.csv",
    index=False
)

val_df.to_csv(
    f"{MANIFEST_DIR}/val_v4.csv",
    index=False
)

test_df.to_csv(
    f"{MANIFEST_DIR}/test_v4.csv",
    index=False
)


# ============================================================
# PART 7 — CUSTOM MULTI-TASK LABEL TRANSFORM
#
# output:
# channel 0 = pancreas region INCLUDING lesion
# channel 1 = lesion
# ============================================================

class ToPancreasLesionChannelsd(
    MapTransform
):

    def __init__(
        self,
        keys
    ):
        super().__init__(keys)


    def __call__(
        self,
        data
    ):

        d = dict(data)

        for key in self.keys:

            x = d[key]

            x = torch.as_tensor(
                x
            )

            x = torch.round(
                x
            ).long()

            pancreas_region = (
                x > 0
            ).float()

            lesion = (
                x == 2
            ).float()

            d[key] = torch.cat(
                [
                    pancreas_region,
                    lesion
                ],
                dim=0
            )

        return d


# ============================================================
# PART 8 — TRANSFORMS
# ============================================================

train_transforms = Compose([

    LoadImaged(
        keys=[
            "image",
            "label"
        ]
    ),

    EnsureChannelFirstd(
        keys=[
            "image",
            "label"
        ]
    ),

    Orientationd(
        keys=[
            "image",
            "label"
        ],
        axcodes="RAS"
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=CT_MIN,
        a_max=CT_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=[
            "image",
            "label"
        ],
        source_key="image",
        margin=8,
        allow_smaller=True
    ),

    Spacingd(
        keys=[
            "image",
            "label"
        ],
        pixdim=SPACING,
        mode=(
            "bilinear",
            "nearest"
        )
    ),

    SpatialPadd(
        keys=[
            "image",
            "label"
        ],
        spatial_size=PATCH
    ),

    RandCropByLabelClassesd(
        keys=[
            "image",
            "label"
        ],
        label_key="label",
        spatial_size=PATCH,
        ratios=CROP_RATIOS,
        num_classes=3,
        num_samples=1,
        allow_smaller=False,
        warn=False
    ),

    ToPancreasLesionChannelsd(
        keys=["label"]
    ),

    RandFlipd(
        keys=[
            "image",
            "label"
        ],
        spatial_axis=0,
        prob=0.30
    ),

    RandFlipd(
        keys=[
            "image",
            "label"
        ],
        spatial_axis=1,
        prob=0.30
    ),

    RandRotate90d(
        keys=[
            "image",
            "label"
        ],
        prob=0.20,
        max_k=3
    ),

    RandShiftIntensityd(
        keys=["image"],
        offsets=0.05,
        prob=0.20
    ),

    EnsureTyped(
        keys=[
            "image",
            "label"
        ]
    ),
])


eval_transforms = Compose([

    LoadImaged(
        keys=[
            "image",
            "label"
        ]
    ),

    EnsureChannelFirstd(
        keys=[
            "image",
            "label"
        ]
    ),

    Orientationd(
        keys=[
            "image",
            "label"
        ],
        axcodes="RAS"
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=CT_MIN,
        a_max=CT_MAX,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=[
            "image",
            "label"
        ],
        source_key="image",
        margin=8,
        allow_smaller=True
    ),

    Spacingd(
        keys=[
            "image",
            "label"
        ],
        pixdim=SPACING,
        mode=(
            "bilinear",
            "nearest"
        )
    ),

    ToPancreasLesionChannelsd(
        keys=["label"]
    ),

    EnsureTyped(
        keys=[
            "image",
            "label"
        ]
    ),
])


# ============================================================
# PART 9 — FILE LISTS
# ============================================================

def dataframe_to_files(
    df
):

    files = []

    for _, row in df.iterrows():

        files.append({

            "image":
                row.image,

            "label":
                row.label,

            "case_id":
                row.case_id,

            "has_lesion":
                int(
                    row.has_lesion
                ),

            "pancreas_valid":
                int(
                    row.pancreas_valid
                ),
        })

    return files


train_files = (
    dataframe_to_files(
        train_df
    )
)

val_files = (
    dataframe_to_files(
        val_df
    )
)

test_files = (
    dataframe_to_files(
        test_df
    )
)


# ============================================================
# PART 10 — DATA LOADERS
# ============================================================

train_ds = Dataset(
    train_files,
    transform=train_transforms
)


sample_weights = [

    POSITIVE_PATIENT_WEIGHT
    if x["has_lesion"] == 1
    else 1.0

    for x in train_files
]


sampler = (
    WeightedRandomSampler(
        sample_weights,
        num_samples=len(
            train_files
        ),
        replacement=True
    )
)


train_loader = DataLoader(
    train_ds,
    batch_size=1,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)


# ------------------------------------------------------------
# FAST VALIDATION:
# all 9 positives + 11 negatives
# ------------------------------------------------------------

val_pos = val_df[
    val_df.has_lesion == 1
]

val_neg = val_df[
    val_df.has_lesion == 0
]

needed = max(
    0,
    20 - len(val_pos)
)

fast_val_df = pd.concat([

    val_pos,

    val_neg.sample(
        n=min(
            needed,
            len(val_neg)
        ),
        random_state=SEED
    )

]).reset_index(drop=True)


fast_val_loader = DataLoader(

    Dataset(
        dataframe_to_files(
            fast_val_df
        ),
        transform=eval_transforms
    ),

    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


val_loader = DataLoader(

    Dataset(
        val_files,
        transform=eval_transforms
    ),

    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


test_loader = DataLoader(

    Dataset(
        test_files,
        transform=eval_transforms
    ),

    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


print(
    "\nTrain iterations:",
    len(train_loader)
)

print(
    "Fast validation:",
    len(fast_val_df)
)


# ============================================================
# PART 11 — DOWNLOAD SuPreM
# ============================================================

if not os.path.exists(
    SUPREM_WEIGHTS
):

    print(
        "\nDownloading SuPreM "
        "SegResNet pretrained weights..."
    )

    download(
        SUPREM_URL,
        SUPREM_WEIGHTS
    )


print(
    "SuPreM checkpoint MB:",
    round(
        os.path.getsize(
            SUPREM_WEIGHTS
        ) / 1024**2,
        1
    )
)


# ============================================================
# PART 12 — MODEL
# ============================================================

torch.cuda.empty_cache()

model = SegResNet(

    spatial_dims=3,

    init_filters=16,

    in_channels=1,

    out_channels=2,

    dropout_prob=0.0,

    blocks_down=(
        1,
        2,
        2,
        4
    ),

    blocks_up=(
        1,
        1,
        1
    ),

).to(device)


total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nV4 parameters:",
    f"{total_params:,}"
)


# ============================================================
# PART 13 — FLEXIBLE SuPreM LOADER
# ============================================================

def load_suprem_weights(
    model,
    path
):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(
        checkpoint,
        dict
    ):

        if "net" in checkpoint:

            source = checkpoint[
                "net"
            ]

        elif "state_dict" in checkpoint:

            source = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            source = checkpoint[
                "model_state_dict"
            ]

        else:

            source = checkpoint

    else:

        source = checkpoint


    target = model.state_dict()

    loaded = {}

    loaded_numel = 0


    for original_key, value in source.items():

        if not torch.is_tensor(
            value
        ):
            continue

        candidates = [
            original_key,
        ]

        parts = (
            original_key.split(".")
        )

        # SuPreM checkpoint commonly has
        # "module." prefix
        if len(parts) > 1:

            candidates.append(
                ".".join(
                    parts[1:]
                )
            )

        if original_key.startswith(
            "module."
        ):

            candidates.append(
                original_key[
                    len("module.") :
                ]
            )

        if original_key.startswith(
            "model."
        ):

            candidates.append(
                original_key[
                    len("model.") :
                ]
            )

        candidates = list(
            dict.fromkeys(
                candidates
            )
        )


        matched = None

        for candidate in candidates:

            if (
                candidate in target
                and
                target[candidate].shape
                ==
                value.shape
            ):

                matched = candidate
                break


        if matched is not None:

            loaded[
                matched
            ] = value

            loaded_numel += (
                value.numel()
            )


    new_state = (
        target.copy()
    )

    new_state.update(
        loaded
    )

    model.load_state_dict(
        new_state
    )


    total_numel = sum(
        x.numel()
        for x in target.values()
    )

    fraction = (
        loaded_numel
        / total_numel
    )


    print(
        "\nSuPreM PRETRAINING LOAD"
    )

    print(
        "Loaded tensors:",
        len(loaded),
        "/",
        len(target)
    )

    print(
        "Loaded parameter fraction:",
        f"{100*fraction:.2f}%"
    )


    # Prevent accidentally training a
    # random network if names do not match.
    if fraction < 0.60:

        raise RuntimeError(
            "Less than 60% of SuPreM "
            "parameters loaded. "
            "STOPPING to avoid random "
            "training."
        )


    return fraction


# ============================================================
# PART 14 — LOSS
# ============================================================

def soft_dice_loss(
    probability,
    target,
    eps=1e-5
):

    dims = tuple(
        range(
            1,
            probability.ndim
        )
    )

    intersection = (
        probability
        * target
    ).sum(
        dim=dims
    )

    denominator = (
        probability.sum(
            dim=dims
        )
        +
        target.sum(
            dim=dims
        )
    )

    dice = (
        (
            2.0
            * intersection
            + eps
        )
        /
        (
            denominator
            + eps
        )
    )

    return 1.0 - dice


def tversky_loss(
    probability,
    target,
    alpha=0.30,
    beta=0.70,
    eps=1e-5
):

    dims = tuple(
        range(
            1,
            probability.ndim
        )
    )

    tp = (
        probability
        * target
    ).sum(
        dim=dims
    )

    fp = (
        probability
        * (1-target)
    ).sum(
        dim=dims
    )

    fn = (
        (1-probability)
        * target
    ).sum(
        dim=dims
    )

    score = (
        tp + eps
    ) / (
        tp
        +
        alpha*fp
        +
        beta*fn
        +
        eps
    )

    return 1.0 - score


def focal_bce(
    logits,
    target,
    gamma=2.0
):

    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none"
        )
    )

    pt = torch.exp(
        -bce
    )

    focal = (
        (1-pt)**gamma
        * bce
    )

    return focal.mean(
        dim=tuple(
            range(
                1,
                focal.ndim
            )
        )
    )


def loss_function(
    logits,
    target,
    pancreas_valid
):

    # ----------------------------------------
    # Channel 0 = pancreas region
    # Channel 1 = lesion
    # ----------------------------------------

    probabilities = (
        torch.sigmoid(
            logits
        )
    )

    pan_prob = probabilities[
        :,
        0:1
    ]

    les_prob = probabilities[
        :,
        1:2
    ]

    pan_target = target[
        :,
        0:1
    ]

    les_target = target[
        :,
        1:2
    ]

    # ----------------------------------------
    # Pancreas
    # ----------------------------------------

    pan_dice = soft_dice_loss(
        pan_prob,
        pan_target
    )

    pan_bce = (
        F.binary_cross_entropy_with_logits(
            logits[:, 0:1],
            pan_target,
            reduction="none"
        )
        .mean(
            dim=(1,2,3,4)
        )
    )


    valid = (
        pancreas_valid
        .float()
        .view(-1)
    )


    if valid.sum() > 0:

        pancreas_loss = (
            (
                0.70*pan_dice
                +
                0.30*pan_bce
            )
            * valid
        ).sum() / valid.sum()

    else:

        pancreas_loss = (
            logits.sum()
            * 0.0
        )


    # ----------------------------------------
    # Lesion
    # ----------------------------------------

    lesion_tversky = (
        tversky_loss(
            les_prob,
            les_target,
            alpha=0.30,
            beta=0.70
        ).mean()
    )

    lesion_focal = (
        focal_bce(
            logits[:, 1:2],
            les_target,
            gamma=2.0
        ).mean()
    )


    # Encourage lesion probability to be
    # inside predicted pancreatic region.

    consistency = (
        torch.relu(
            les_prob
            -
            pan_prob
        )
        .mean()
    )


    total = (

        0.75
        * pancreas_loss

        +

        2.00
        * lesion_tversky

        +

        0.50
        * lesion_focal

        +

        0.10
        * consistency
    )

    return total


# ============================================================
# PART 15 — OPTIMIZER
#
# Smaller LR for pretrained body.
# Higher LR for new final prediction head.
# ============================================================

head_params = []

base_params = []

for name, parameter in (
    model.named_parameters()
):

    if "conv_final" in name:

        head_params.append(
            parameter
        )

    else:

        base_params.append(
            parameter
        )


optimizer = torch.optim.AdamW([

    {
        "params":
            base_params,

        "lr":
            BASE_LR,
    },

    {
        "params":
            head_params,

        "lr":
            HEAD_LR,
    },

],
    weight_decay=1e-5
)


scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=1e-6
    )
)


scaler = (
    torch.amp.GradScaler(
        "cuda"
    )
)


# ============================================================
# PART 16 — METRIC FUNCTIONS
# ============================================================

THRESHOLDS = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]


def threshold_key(t):

    return str(t).replace(
        ".",
        "p"
    )


def binary_dice(
    pred,
    gt
):

    pred = pred.astype(
        bool
    )

    gt = gt.astype(
        bool
    )

    denominator = (
        pred.sum()
        +
        gt.sum()
    )

    if denominator == 0:

        return np.nan

    return float(
        2
        * np.logical_and(
            pred,
            gt
        ).sum()
        /
        denominator
    )


def largest_component(
    mask
):

    mask = mask.astype(
        bool
    )

    if not mask.any():

        return 0

    labeled, n = (
        ndimage.label(
            mask
        )
    )

    if n == 0:

        return 0

    sizes = np.bincount(
        labeled.ravel()
    )

    if len(sizes) <= 1:

        return 0

    return int(
        sizes[1:].max()
    )


def get_case_id(
    batch
):

    value = batch[
        "case_id"
    ]

    if isinstance(
        value,
        (
            list,
            tuple
        )
    ):

        return str(
            value[0]
        )

    return str(
        value
    )


# ============================================================
# PART 17 — EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    name="Validation"
):

    model.eval()

    rows = []


    with torch.inference_mode():

        for batch in tqdm(
            loader,
            desc=name,
            leave=False
        ):

            image = (
                batch["image"]
                .to(
                    device,
                    non_blocking=True
                )
            )

            label = (
                batch["label"]
                .float()
                .to(
                    device,
                    non_blocking=True
                )
            )


            with torch.amp.autocast(
                device_type="cuda"
            ):

                logits = (
                    sliding_window_inference(
                        image,
                        roi_size=PATCH,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.50,
                        mode="gaussian"
                    )
                )


            prob = (
                torch.sigmoid(
                    logits
                )[0]
                .float()
                .cpu()
                .numpy()
            )


            gt = (
                label[0]
                .cpu()
                .numpy()
            )


            pan_prob = prob[0]

            lesion_prob = prob[1]

            gt_pan = (
                gt[0] > 0.5
            )

            gt_lesion = (
                gt[1] > 0.5
            )


            pancreas_valid = int(
                torch.as_tensor(
                    batch[
                        "pancreas_valid"
                    ]
                )
                .flatten()[0]
                .item()
            )


            # --------------------------------
            # Pancreas Dice
            # --------------------------------

            if pancreas_valid:

                pred_pan = (
                    pan_prob > 0.5
                )

                pan_dice = (
                    binary_dice(
                        pred_pan,
                        gt_pan
                    )
                )

            else:

                pan_dice = np.nan


            # --------------------------------
            # Patient-level continuous score
            #
            # Robust high percentile inside
            # likely pancreas region.
            # --------------------------------

            roi = (
                pan_prob > 0.20
            )

            if roi.any():

                patient_score = float(
                    np.quantile(
                        lesion_prob[
                            roi
                        ],
                        0.995
                    )
                )

            else:

                patient_score = float(
                    np.quantile(
                        lesion_prob,
                        0.995
                    )
                )


            row = {

                "case_id":
                    get_case_id(
                        batch
                    ),

                "gt_positive":
                    bool(
                        gt_lesion.any()
                    ),

                "pancreas_valid":
                    pancreas_valid,

                "pancreas_dice":
                    pan_dice,

                "patient_score":
                    patient_score,

                "gt_lesion_voxels":
                    int(
                        gt_lesion.sum()
                    ),
            }


            # --------------------------------
            # Threshold audit
            # --------------------------------

            for threshold in (
                THRESHOLDS
            ):

                key = (
                    threshold_key(
                        threshold
                    )
                )

                # Anatomy gate:
                # lesion must overlap likely
                # pancreas region.

                pred_lesion = (

                    (
                        lesion_prob
                        >
                        threshold
                    )

                    &

                    (
                        pan_prob
                        >
                        0.20
                    )
                )


                row[
                    f"pred_voxels_{key}"
                ] = int(
                    pred_lesion.sum()
                )


                row[
                    f"largest_component_{key}"
                ] = (
                    largest_component(
                        pred_lesion
                    )
                )


                row[
                    f"lesion_dice_{key}"
                ] = (
                    binary_dice(
                        pred_lesion,
                        gt_lesion
                    )
                    if gt_lesion.any()
                    else np.nan
                )


            rows.append(
                row
            )


    df = pd.DataFrame(
        rows
    )


    # ----------------------------------------
    # Pancreas Dice
    # ----------------------------------------

    mean_pancreas = (
        df.pancreas_dice
        .dropna()
        .mean()
    )


    # ----------------------------------------
    # AUC
    # ----------------------------------------

    try:

        auc = roc_auc_score(
            df.gt_positive.astype(
                int
            ),
            df.patient_score
        )

    except Exception:

        auc = np.nan


    # ----------------------------------------
    # Mean lesion Dice for each threshold
    # ----------------------------------------

    threshold_dices = {}

    positives = df[
        df.gt_positive
    ]


    for threshold in (
        THRESHOLDS
    ):

        key = threshold_key(
            threshold
        )

        value = (
            positives[
                f"lesion_dice_{key}"
            ]
            .dropna()
            .mean()
        )

        threshold_dices[
            threshold
        ] = (
            0.0
            if np.isnan(value)
            else float(value)
        )


    best_threshold = max(
        threshold_dices,
        key=threshold_dices.get
    )

    best_lesion_dice = (
        threshold_dices[
            best_threshold
        ]
    )


    metrics = {

        "n":
            len(df),

        "positive_cases":
            int(
                df.gt_positive.sum()
            ),

        "pancreas_dice":
            float(
                0
                if np.isnan(
                    mean_pancreas
                )
                else mean_pancreas
            ),

        "auc":
            (
                None
                if np.isnan(auc)
                else float(auc)
            ),

        "best_lesion_threshold":
            float(
                best_threshold
            ),

        "best_positive_lesion_dice":
            float(
                best_lesion_dice
            ),

        "threshold_dices":
            {
                str(k):
                    float(v)
                for k, v
                in threshold_dices.items()
            },
    }


    return metrics, df


# ============================================================
# PART 18 — PRETRAIN / RESUME
# ============================================================

history = []

start_epoch = 1

best_score = -1.0

best_lesion_seen = 0.0

epochs_without_improvement = 0


if os.path.exists(
    HISTORY_FILE
):

    try:

        history = (
            pd.read_csv(
                HISTORY_FILE
            )
            .to_dict(
                "records"
            )
        )

    except Exception:

        history = []


if os.path.exists(
    LAST_MODEL
):

    print(
        "\nV4 checkpoint found."
    )

    print(
        "RESUMING V4..."
    )

    checkpoint = torch.load(
        LAST_MODEL,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    scaler.load_state_dict(
        checkpoint[
            "scaler_state_dict"
        ]
    )

    start_epoch = int(
        checkpoint["epoch"]
    ) + 1

    best_score = float(
        checkpoint.get(
            "best_score",
            -1
        )
    )

    best_lesion_seen = float(
        checkpoint.get(
            "best_lesion_seen",
            0
        )
    )

    epochs_without_improvement = int(
        checkpoint.get(
            "epochs_without_improvement",
            0
        )
    )


else:

    pretrained_fraction = (
        load_suprem_weights(
            model,
            SUPREM_WEIGHTS
        )
    )

    print(
        "\nSuPreM initialized successfully."
    )


# ============================================================
# PART 19 — GPU SMOKE TEST
# ============================================================

print(
    "\nRunning V4 GPU smoke test..."
)

model.eval()

with torch.inference_mode():

    x = torch.randn(
        1,
        1,
        *PATCH,
        device=device
    )

    with torch.amp.autocast(
        device_type="cuda"
    ):

        y = model(x)


print(
    "Input:",
    tuple(x.shape)
)

print(
    "Output:",
    tuple(y.shape)
)

assert tuple(y.shape) == (
    1,
    2,
    *PATCH
)

del x, y

torch.cuda.empty_cache()

print(
    "V4 SMOKE TEST PASSED"
)


# ============================================================
# PART 20 — TRAINING
# ============================================================

print(
    "\n" + "="*65
)

print(
    "V4 SuPreM TRAINING STARTED"
)

print(
    "="*65
)

print(
    "Starting epoch:",
    start_epoch
)


training_start = (
    time.time()
)


for epoch in range(
    start_epoch,
    MAX_EPOCHS + 1
):

    epoch_start = (
        time.time()
    )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0


    progress = tqdm(
        train_loader,
        desc=(
            f"V4 Epoch "
            f"{epoch}/"
            f"{MAX_EPOCHS}"
        )
    )


    for step, batch in enumerate(
        progress,
        start=1
    ):

        image = (
            batch["image"]
            .to(
                device,
                non_blocking=True
            )
        )

        label = (
            batch["label"]
            .float()
            .to(
                device,
                non_blocking=True
            )
        )

        pancreas_valid = (
            torch.as_tensor(
                batch[
                    "pancreas_valid"
                ]
            )
            .to(device)
        )


        with torch.amp.autocast(
            device_type="cuda"
        ):

            logits = model(
                image
            )

            loss = (
                loss_function(
                    logits,
                    label,
                    pancreas_valid
                )
                /
                GRAD_ACCUM
            )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                "Non-finite loss."
            )


        scaler.scale(
            loss
        ).backward()


        if (
            step % GRAD_ACCUM == 0
            or
            step == len(
                train_loader
            )
        ):

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )


        actual_loss = (
            loss.item()
            * GRAD_ACCUM
        )

        running_loss += (
            actual_loss
        )


        progress.set_postfix({

            "loss":
                f"{actual_loss:.4f}",

            "base_lr":
                f"{optimizer.param_groups[0]['lr']:.1e}"
        })


    scheduler.step()


    train_loss = (
        running_loss
        /
        len(train_loader)
    )


    # ========================================================
    # FAST VALIDATION
    # ========================================================

    fast_metrics, fast_results = (
        evaluate(
            model,
            fast_val_loader,
            name=(
                f"Fast Val E{epoch}"
            )
        )
    )


    lesion_dice = float(
        fast_metrics[
            "best_positive_lesion_dice"
        ]
    )

    pancreas_dice = float(
        fast_metrics[
            "pancreas_dice"
        ]
    )

    auc = (
        fast_metrics[
            "auc"
        ]
    )

    auc_value = (
        0.5
        if auc is None
        else auc
    )


    # Primary metric = lesion localization.
    # Pancreas and patient AUC are secondary.

    selection_score = (

        0.75
        * lesion_dice

        +

        0.15
        * pancreas_dice

        +

        0.10
        * max(
            auc_value - 0.5,
            0.0
        )
    )


    elapsed_minutes = (
        time.time()
        - epoch_start
    ) / 60


    print(
        "\n" + "-"*60
    )

    print(
        f"V4 EPOCH {epoch}"
    )

    print(
        "Train Loss:",
        f"{train_loss:.4f}"
    )

    print(
        "Fast Pancreas Dice:",
        f"{pancreas_dice:.4f}"
    )

    print(
        "Fast BEST Lesion Dice:",
        f"{lesion_dice:.4f}"
    )

    print(
        "Best lesion threshold:",
        fast_metrics[
            "best_lesion_threshold"
        ]
    )

    print(
        "Fast AUC:",
        auc
    )

    print(
        "Selection Score:",
        f"{selection_score:.4f}"
    )

    print(
        "Epoch minutes:",
        f"{elapsed_minutes:.1f}"
    )


    history.append({

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "fast_pancreas_dice":
            pancreas_dice,

        "fast_lesion_dice":
            lesion_dice,

        "fast_best_threshold":
            fast_metrics[
                "best_lesion_threshold"
            ],

        "fast_auc":
            auc,

        "selection_score":
            selection_score,

        "epoch_minutes":
            elapsed_minutes,

        "base_lr":
            optimizer.param_groups[
                0
            ]["lr"],

        "head_lr":
            optimizer.param_groups[
                1
            ]["lr"],
    })


    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_FILE,
        index=False
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if selection_score > (
        best_score + 1e-5
    ):

        best_score = (
            selection_score
        )

        best_lesion_seen = max(
            best_lesion_seen,
            lesion_dice
        )

        epochs_without_improvement = 0


        torch.save({

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "selection_score":
                selection_score,

            "best_score":
                best_score,

            "best_lesion_seen":
                best_lesion_seen,

            "fast_metrics":
                fast_metrics,

            "architecture":
                "MONAI SegResNet",

            "pretraining":
                "SuPreM SegResNet 2100 CT",

            "spacing":
                SPACING,

            "patch":
                PATCH,

            "ct_window":
                [
                    CT_MIN,
                    CT_MAX
                ],

        }, BEST_MODEL)


        print(
            ">>> BEST V4 MODEL SAVED"
        )


    else:

        epochs_without_improvement += 1


    # ========================================================
    # LAST CHECKPOINT — EVERY EPOCH
    # ========================================================

    torch.save({

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "best_score":
            best_score,

        "best_lesion_seen":
            best_lesion_seen,

        "epochs_without_improvement":
            epochs_without_improvement,

    }, LAST_MODEL)


    print(
        "Last checkpoint saved."
    )


    # ========================================================
    # EARLY STOP
    # ========================================================

    if (
        epoch >= 8
        and
        epochs_without_improvement >= 4
    ):

        print(
            "\nEarly stopping:"
            " no improvement for "
            "4 epochs."
        )

        break


training_hours = (
    time.time()
    - training_start
) / 3600


print(
    "\nTraining complete."
)

print(
    "Training hours:",
    round(
        training_hours,
        2
    )
)


# ============================================================
# PART 21 — LOAD BEST
# ============================================================

assert os.path.exists(
    BEST_MODEL
)

best_checkpoint = torch.load(
    BEST_MODEL,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()


print(
    "\nLoaded best V4 epoch:",
    best_checkpoint[
        "epoch"
    ]
)


# ============================================================
# PART 22 — FULL 100-CASE VALIDATION
# ============================================================

print(
    "\nRunning FULL "
    "100-case validation..."
)

val_metrics, val_results = (
    evaluate(
        model,
        val_loader,
        name="Full Validation"
    )
)


val_results.to_csv(
    f"{RESULTS_DIR}/"
    "v4_validation_cases.csv",
    index=False
)


print(
    "\nVALIDATION METRICS"
)

print(
    json.dumps(
        val_metrics,
        indent=2
    )
)


# ============================================================
# PART 23 — SELECT LESION THRESHOLD USING VALIDATION ONLY
# ============================================================

BEST_LESION_THRESHOLD = float(
    val_metrics[
        "best_lesion_threshold"
    ]
)

BEST_KEY = threshold_key(
    BEST_LESION_THRESHOLD
)


# ============================================================
# PART 24 — SELECT PATIENT COMPONENT THRESHOLD USING VAL ONLY
# ============================================================

COMPONENT_THRESHOLDS = [
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
]


def detection_metrics(
    df,
    lesion_threshold,
    component_threshold
):

    key = threshold_key(
        lesion_threshold
    )

    prediction = (

        df[
            f"largest_component_{key}"
        ]

        >=

        component_threshold
    )


    truth = (
        df[
            "gt_positive"
        ].astype(bool)
    )


    tp = int(
        (
            prediction
            &
            truth
        ).sum()
    )

    tn = int(
        (
            (~prediction)
            &
            (~truth)
        ).sum()
    )

    fp = int(
        (
            prediction
            &
            (~truth)
        ).sum()
    )

    fn = int(
        (
            (~prediction)
            &
            truth
        ).sum()
    )


    sensitivity = (
        tp
        /
        (
            tp
            +
            fn
            +
            1e-8
        )
    )

    specificity = (
        tn
        /
        (
            tn
            +
            fp
            +
            1e-8
        )
    )

    precision = (
        tp
        /
        (
            tp
            +
            fp
            +
            1e-8
        )
    )

    balanced_accuracy = (
        sensitivity
        +
        specificity
    ) / 2


    return {

        "lesion_threshold":
            lesion_threshold,

        "component_threshold":
            component_threshold,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision,

        "balanced_accuracy":
            balanced_accuracy,

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,
    }


component_search = []

for component_threshold in (
    COMPONENT_THRESHOLDS
):

    component_search.append(

        detection_metrics(

            val_results,

            BEST_LESION_THRESHOLD,

            component_threshold
        )
    )


component_df = pd.DataFrame(
    component_search
)


component_df.to_csv(
    f"{RESULTS_DIR}/"
    "v4_component_search.csv",
    index=False
)


best_detection_row = (

    component_df

    .sort_values(

        [
            "balanced_accuracy",
            "sensitivity",
            "specificity"
        ],

        ascending=[
            False,
            False,
            False
        ]
    )

    .iloc[0]
)


BEST_COMPONENT_THRESHOLD = int(

    best_detection_row[
        "component_threshold"
    ]
)


print(
    "\nVALIDATION-SELECTED PARAMETERS"
)

print(
    "Lesion probability threshold:",
    BEST_LESION_THRESHOLD
)

print(
    "Component threshold:",
    BEST_COMPONENT_THRESHOLD
)

print(
    best_detection_row
)


# ============================================================
# PART 25 — UNTOUCHED LOCAL TEST
# ============================================================

print(
    "\nRunning UNTOUCHED "
    "100-case local test..."
)


test_metrics, test_results = (
    evaluate(
        model,
        test_loader,
        name="Local Test"
    )
)


test_results.to_csv(
    f"{RESULTS_DIR}/"
    "v4_test_cases.csv",
    index=False
)


test_detection = detection_metrics(

    test_results,

    BEST_LESION_THRESHOLD,

    BEST_COMPONENT_THRESHOLD
)


# Fixed validation-selected Dice

test_dice_col = (
    f"lesion_dice_"
    f"{BEST_KEY}"
)

test_positive_dice = float(

    test_results.loc[
        test_results.gt_positive,
        test_dice_col
    ]
    .dropna()
    .mean()
)


print(
    "\n" + "="*65
)

print(
    "V4 LOCAL TEST RESULTS"
)

print(
    "="*65
)

print(
    "Positive Lesion Dice:",
    test_positive_dice
)

print(
    "Pancreas Dice:",
    test_metrics[
        "pancreas_dice"
    ]
)

print(
    "Patient AUC:",
    test_metrics[
        "auc"
    ]
)

print(
    "Sensitivity:",
    test_detection[
        "sensitivity"
    ]
)

print(
    "Specificity:",
    test_detection[
        "specificity"
    ]
)

print(
    "Precision:",
    test_detection[
        "precision"
    ]
)

print(
    "TP/TN/FP/FN:",
    test_detection["TP"],
    test_detection["TN"],
    test_detection["FP"],
    test_detection["FN"]
)




# ============================================================
# PART 26 — OFFICIAL 901-CASE TEST (RESUMABLE)
# ============================================================

OFFICIAL_IMAGE_ROOT = "/content/PanTS/data/ImageTe"
OFFICIAL_SEG_ROOT = "/content/PanTS/data/LabelTe"
OFFICIAL_LABEL3_ROOT = "/content/PanTS/v4_work/Label3Te"

DRIVE_DOWNLOADS = "/content/drive/MyDrive/PanTS_Project/downloads"
OFFICIAL_IMAGE_ARCHIVE = (
    f"{DRIVE_DOWNLOADS}/"
    "PanTSMini_ImageTe_00009001_00009901.tar.gz"
)
OFFICIAL_LABEL_ARCHIVE = (
    f"{DRIVE_DOWNLOADS}/PanTS_Test_Labels_901.tar.gz"
)

OFFICIAL_CASE_CSV = f"{RESULTS_DIR}/official_901_cases.csv"
OFFICIAL_SUMMARY_JSON = f"{RESULTS_DIR}/official_901_summary.json"

os.makedirs(OFFICIAL_IMAGE_ROOT, exist_ok=True)
os.makedirs(OFFICIAL_SEG_ROOT, exist_ok=True)
os.makedirs(OFFICIAL_LABEL3_ROOT, exist_ok=True)


def count_official_ct():
    return len(glob.glob(f"{OFFICIAL_IMAGE_ROOT}/PanTS_*/ct.nii.gz"))


def count_official_seg(name):
    return len(glob.glob(
        f"{OFFICIAL_SEG_ROOT}/PanTS_*/segmentations/{name}.nii.gz"
    ))


# Restore official images from the verified Drive archive if Colab reset.
if count_official_ct() != 901:
    if not os.path.exists(OFFICIAL_IMAGE_ARCHIVE):
        raise FileNotFoundError(
            "Official 901-case image archive is missing from Google Drive: "
            + OFFICIAL_IMAGE_ARCHIVE
        )
    print("\nRestoring official 901 test CTs from Google Drive...")
    run([
        "tar", "-xzf", OFFICIAL_IMAGE_ARCHIVE,
        "-C", OFFICIAL_IMAGE_ROOT
    ])

assert count_official_ct() == 901, count_official_ct()


# Restore the compact 901-label backup if needed.
if (
    count_official_seg("pancreas") != 901
    or count_official_seg("pancreatic_lesion") != 901
):
    if not os.path.exists(OFFICIAL_LABEL_ARCHIVE):
        raise FileNotFoundError(
            "Official 901-case label archive is missing from Google Drive: "
            + OFFICIAL_LABEL_ARCHIVE
        )
    print("\nRestoring official 901 test labels from Google Drive...")
    run([
        "tar", "-xzf", OFFICIAL_LABEL_ARCHIVE,
        "-C", "/content/PanTS/data"
    ])

assert count_official_seg("pancreas") == 901
assert count_official_seg("pancreatic_lesion") == 901


# Build 3-class labels matching the training representation:
# 0 background, 1 pancreas, 2 lesion.
official_rows = []

print("\nPreparing official 901-case labels...")
for i in tqdm(range(9001, 9902), desc="Official labels"):
    case = f"PanTS_{i:08d}"
    ct_path = f"{OFFICIAL_IMAGE_ROOT}/{case}/ct.nii.gz"
    pan_path = (
        f"{OFFICIAL_SEG_ROOT}/{case}/segmentations/pancreas.nii.gz"
    )
    les_path = (
        f"{OFFICIAL_SEG_ROOT}/{case}/segmentations/pancreatic_lesion.nii.gz"
    )
    out_dir = f"{OFFICIAL_LABEL3_ROOT}/{case}"
    os.makedirs(out_dir, exist_ok=True)
    label_path = f"{out_dir}/label_3class.nii.gz"

    if not os.path.exists(label_path):
        ct_nii = nib.load(ct_path)
        pan_nii = nib.load(pan_path)
        les_nii = nib.load(les_path)

        pan = np.asarray(pan_nii.dataobj) > 0
        les = np.asarray(les_nii.dataobj) > 0

        if pan.shape != les.shape or pan.shape != ct_nii.shape:
            raise RuntimeError(f"Shape mismatch in {case}")

        # Require matching geometry; tiny floating-point differences are OK.
        if not np.allclose(ct_nii.affine, pan_nii.affine, atol=1e-4):
            raise RuntimeError(f"CT/pancreas affine mismatch in {case}")
        if not np.allclose(ct_nii.affine, les_nii.affine, atol=1e-4):
            raise RuntimeError(f"CT/lesion affine mismatch in {case}")

        target = np.zeros(pan.shape, dtype=np.uint8)
        target[pan] = 1
        target[les] = 2

        out = nib.Nifti1Image(
            target,
            affine=ct_nii.affine,
            header=ct_nii.header.copy(),
        )
        out.set_data_dtype(np.uint8)
        nib.save(out, label_path)

    mask = np.asarray(nib.load(label_path).dataobj)
    official_rows.append({
        "case_id": case,
        "image": ct_path,
        "label": label_path,
        "pancreas_valid": int((mask > 0).any()),
        "has_lesion": int((mask == 2).any()),
    })


official_df = pd.DataFrame(official_rows)
official_df.to_csv(
    f"{MANIFEST_DIR}/official_test_901_manifest.csv",
    index=False,
)

print("Official cases:", len(official_df))
print("Official lesion-positive:", int(official_df.has_lesion.sum()))
print("Official lesion-negative:", int((official_df.has_lesion == 0).sum()))


def remove_small_components(mask, min_size):
    mask = mask.astype(bool)
    if min_size <= 1 or not mask.any():
        return mask
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_size)[0]
    keep = keep[keep != 0]
    if len(keep) == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(lab, keep)


def tumor_detection_counts(gt_lesion, pred_lesion):
    """Count GT connected lesions and how many overlap a prediction."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    lab, n = ndimage.label(gt_lesion.astype(bool), structure=structure)
    detected = 0
    for idx in range(1, n + 1):
        if np.logical_and(lab == idx, pred_lesion).any():
            detected += 1
    return int(n), int(detected)


def official_inference_row(batch):
    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].float().to(device, non_blocking=True)

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
        logits = sliding_window_inference(
            image,
            roi_size=PATCH,
            sw_batch_size=1,
            predictor=model,
            overlap=0.50,
            mode="gaussian",
        )

    prob = torch.sigmoid(logits)[0].float().cpu().numpy()
    gt = label[0].cpu().numpy()

    pan_prob = prob[0]
    lesion_prob = prob[1]
    gt_pan = gt[0] > 0.5
    gt_lesion = gt[1] > 0.5

    pred_pan = pan_prob > 0.5
    pred_lesion_raw = (
        (lesion_prob > BEST_LESION_THRESHOLD)
        & (pan_prob > 0.20)
    )
    pred_lesion = remove_small_components(
        pred_lesion_raw,
        BEST_COMPONENT_THRESHOLD,
    )

    roi = pan_prob > 0.20
    if roi.any():
        patient_score = float(np.quantile(lesion_prob[roi], 0.995))
    else:
        patient_score = float(np.quantile(lesion_prob, 0.995))

    tumor_total, tumor_detected = tumor_detection_counts(
        gt_lesion,
        pred_lesion,
    )

    return {
        "case_id": get_case_id(batch),
        "gt_positive": bool(gt_lesion.any()),
        "pred_positive": bool(pred_lesion.any()),
        "patient_score": patient_score,
        "pancreas_dice": binary_dice(pred_pan, gt_pan),
        "lesion_dice": (
            binary_dice(pred_lesion, gt_lesion)
            if gt_lesion.any() else np.nan
        ),
        "gt_lesion_voxels": int(gt_lesion.sum()),
        "pred_lesion_voxels": int(pred_lesion.sum()),
        "largest_pred_component": largest_component(pred_lesion),
        "tumors_total": tumor_total,
        "tumors_detected": tumor_detected,
    }


# Resume case-by-case official evaluation after a Colab interruption.
if os.path.exists(OFFICIAL_CASE_CSV):
    try:
        completed_df = pd.read_csv(OFFICIAL_CASE_CSV)
        completed_ids = set(completed_df.case_id.astype(str))
        official_result_rows = completed_df.to_dict("records")
        print("\nResuming official test. Completed:", len(completed_ids))
    except Exception:
        completed_ids = set()
        official_result_rows = []
else:
    completed_ids = set()
    official_result_rows = []

remaining_df = official_df[
    ~official_df.case_id.isin(completed_ids)
].reset_index(drop=True)

remaining_loader = DataLoader(
    Dataset(
        dataframe_to_files(remaining_df),
        transform=eval_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

print("Official cases remaining:", len(remaining_df))

model.eval()
for batch in tqdm(remaining_loader, desc="OFFICIAL 901 TEST"):
    row = official_inference_row(batch)
    official_result_rows.append(row)

    # Persist after every case so an interrupted Colab session can resume.
    pd.DataFrame(official_result_rows).to_csv(
        OFFICIAL_CASE_CSV,
        index=False,
    )

    # Keep GPU memory stable for long runs.
    del batch
    torch.cuda.empty_cache()


official_results = pd.DataFrame(official_result_rows)
official_results = (
    official_results
    .drop_duplicates(subset=["case_id"], keep="last")
    .sort_values("case_id")
    .reset_index(drop=True)
)
official_results.to_csv(OFFICIAL_CASE_CSV, index=False)

if len(official_results) != 901:
    raise RuntimeError(
        f"Official evaluation incomplete: {len(official_results)}/901"
    )

truth = official_results.gt_positive.astype(bool)
pred = official_results.pred_positive.astype(bool)

tp = int((truth & pred).sum())
tn = int((~truth & ~pred).sum())
fp = int((~truth & pred).sum())
fn = int((truth & ~pred).sum())

p_sen = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
precision = tp / (tp + fp + 1e-8)

try:
    official_auc = float(roc_auc_score(
        truth.astype(int),
        official_results.patient_score,
    ))
except Exception:
    official_auc = None

tumors_total = int(official_results.tumors_total.sum())
tumors_detected = int(official_results.tumors_detected.sum())
t_sen = tumors_detected / (tumors_total + 1e-8)

positive_dice = float(
    official_results.loc[truth, "lesion_dice"].dropna().mean()
)
mean_pancreas_dice = float(
    official_results.pancreas_dice.dropna().mean()
)

official_summary = {
    "n_cases": 901,
    "positive_cases": int(truth.sum()),
    "negative_cases": int((~truth).sum()),
    "validation_selected_lesion_threshold": float(BEST_LESION_THRESHOLD),
    "validation_selected_component_threshold": int(BEST_COMPONENT_THRESHOLD),
    "P_Sen_patient_sensitivity": float(p_sen),
    "T_Sen_tumor_sensitivity": float(t_sen),
    "specificity": float(specificity),
    "precision": float(precision),
    "AUC": official_auc,
    "DSC_positive_cases": positive_dice,
    "pancreas_DSC": mean_pancreas_dice,
    "TP": tp,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "GT_tumors": tumors_total,
    "detected_GT_tumors": tumors_detected,
}

with open(OFFICIAL_SUMMARY_JSON, "w") as f:
    json.dump(official_summary, f, indent=2)

print("\n" + "=" * 65)
print("OFFICIAL 901-CASE RESULTS")
print("=" * 65)
print(json.dumps(official_summary, indent=2))


# ============================================================
# PART 27 — SAVE FINAL SUMMARY / CONFIG / SUBMISSION PACKAGE
# ============================================================

summary = {
    "model": "SuPreM pretrained MONAI SegResNet",
    "training_subset": "PanTSMini cases 00000001-00001000",
    "training_cases": int(len(train_df)),
    "validation_cases": int(len(val_df)),
    "local_test_cases": int(len(test_df)),
    "pretraining": "SuPreM SegResNet checkpoint pretrained on abdominal CT",
    "spacing": list(SPACING),
    "patch": list(PATCH),
    "ct_window": [CT_MIN, CT_MAX],
    "best_epoch": int(best_checkpoint["epoch"]),
    "validation_metrics": val_metrics,
    "selected_lesion_threshold": float(BEST_LESION_THRESHOLD),
    "selected_component_threshold": int(BEST_COMPONENT_THRESHOLD),
    "local_test_positive_lesion_dice": float(test_positive_dice),
    "local_test_pancreas_dice": float(test_metrics["pancreas_dice"]),
    "local_test_auc": test_metrics["auc"],
    "local_test_detection": test_detection,
    "official_901": official_summary,
    "training_hours_this_run": float(training_hours),
}

with open(f"{RESULTS_DIR}/v4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

config = {
    "seed": SEED,
    "epochs": MAX_EPOCHS,
    "patch": list(PATCH),
    "spacing": list(SPACING),
    "ct_min": CT_MIN,
    "ct_max": CT_MAX,
    "crop_ratios": CROP_RATIOS,
    "positive_patient_weight": POSITIVE_PATIENT_WEIGHT,
    "base_lr": BASE_LR,
    "head_lr": HEAD_LR,
    "gradient_accumulation": GRAD_ACCUM,
}
with open(f"{RESULTS_DIR}/v4_config.json", "w") as f:
    json.dump(config, f, indent=2)

readme = f"""# PanTS Technical Screening — V4 SuPreM SegResNet

## Model
MONAI SegResNet initialized from the publicly released SuPreM abdominal-CT checkpoint.

## Training subset
The model was fine-tuned on the first 1,000 PanTSMini training cases, using a reproducible 800/100/100 train/validation/local-test split. This is a subset of the full PanTS training cohort and must be disclosed.

## Targets
- Output channel 0: pancreas region (including lesion)
- Output channel 1: pancreatic lesion

## Preprocessing
- Orientation: RAS
- Spacing: {SPACING}
- CT window: [{CT_MIN}, {CT_MAX}] HU
- Training patch: {PATCH}

## Best checkpoint
Epoch: {int(best_checkpoint['epoch'])}

## Validation-selected inference parameters
- Lesion probability threshold: {BEST_LESION_THRESHOLD}
- Minimum connected-component size: {BEST_COMPONENT_THRESHOLD}

## Local 100-case held-out test
- Positive-case lesion Dice: {test_positive_dice:.6f}
- Pancreas Dice: {float(test_metrics['pancreas_dice']):.6f}
- Patient AUC: {test_metrics['auc']}
- Sensitivity: {test_detection['sensitivity']:.6f}
- Specificity: {test_detection['specificity']:.6f}

## Official 901-case PanTS in-distribution test
- P-Sen: {official_summary['P_Sen_patient_sensitivity']:.6f}
- T-Sen: {official_summary['T_Sen_tumor_sensitivity']:.6f}
- Specificity: {official_summary['specificity']:.6f}
- AUC: {official_summary['AUC']}
- Positive-case lesion DSC: {official_summary['DSC_positive_cases']:.6f}
- Pancreas DSC: {official_summary['pancreas_DSC']:.6f}

## Reproducibility
Checkpoints, per-case official results, manifests, training history and configuration are included in this package.
"""

with open(f"{SUBMISSION_DIR}/README.md", "w") as f:
    f.write(readme)

# Copy key artifacts into submission directory.
for src, dst_name in [
    (BEST_MODEL, "best_v4_suprem_segresnet.pt"),
    (LAST_MODEL, "last_v4_suprem_segresnet.pt"),
    (HISTORY_FILE, "training_history.csv"),
    (f"{RESULTS_DIR}/v4_summary.json", "v4_summary.json"),
    (OFFICIAL_SUMMARY_JSON, "official_901_summary.json"),
    (OFFICIAL_CASE_CSV, "official_901_cases.csv"),
    (f"{RESULTS_DIR}/v4_validation_cases.csv", "v4_validation_cases.csv"),
    (f"{RESULTS_DIR}/v4_test_cases.csv", "v4_local_test_cases.csv"),
    (f"{MANIFEST_DIR}/train_v4.csv", "train_manifest.csv"),
    (f"{MANIFEST_DIR}/val_v4.csv", "validation_manifest.csv"),
    (f"{MANIFEST_DIR}/test_v4.csv", "local_test_manifest.csv"),
    (f"{MANIFEST_DIR}/official_test_901_manifest.csv", "official_test_901_manifest.csv"),
]:
    if os.path.exists(src):
        shutil.copy2(src, f"{SUBMISSION_DIR}/{dst_name}")

# Copy this script itself if the path exists.
try:
    this_script = os.path.abspath(__file__)
    if os.path.exists(this_script):
        shutil.copy2(this_script, f"{SUBMISSION_DIR}/PanTS_V4_Colab_AllInOne.py")
except Exception:
    pass

ZIP_BASE = f"{PROJECT}/Taukir_Alam_PanTS_V4_SuPreM_Submission"
if os.path.exists(ZIP_BASE + ".zip"):
    os.remove(ZIP_BASE + ".zip")
shutil.make_archive(ZIP_BASE, "zip", SUBMISSION_DIR)

print("\n" + "=" * 65)
print("ALL TASKS FINISHED")
print("=" * 65)
print("BEST MODEL:", BEST_MODEL)
print("OFFICIAL RESULTS:", OFFICIAL_SUMMARY_JSON)
print("SUBMISSION ZIP:", ZIP_BASE + ".zip")
print("\nEverything important is saved under Google Drive:")
print(PROJECT)
