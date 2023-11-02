from pathlib import Path

ROOT = Path("/home/cmti/jogi/diag_plz/med-flamingo")
    
IMAGE_PATH = {
    "kstr": ROOT / "data" / "kstr" / "new_images"
}

FILTERED_CSV = {
    "kstr": ROOT / "data" / "kstr" / "kstr_data_filtered.csv"
}

INSTRUCTION = {
    "few": (
        "You are a helpful medical assistant. "
        "You are being provided with some images, a medical history about the patient, some image findings, and a final diagnosis. "
        f"Follow the examples and provide the final diagnosis for the last case."
    ),
    "zero": (
        "You are a helpful medical assistant. "
        "You are being provided with some images, a medical history about the patient, and some image findings. "
        f"Provide the final diagnosis in short text. Do not provide anything else, such as a discussion or an explanation."
    )
}

FEW_SHOT_SUPPORT_SET = [
    {
        "id": 1,
        "img_count": 5,
        "img_dir": IMAGE_PATH["kstr"] / "case_1342", # for testing
        "history": "sample history",
        "findings": "sample findings",
        "diagnosis": "sample diagnosis"
    }
]

MAX_SHOTS = len(FEW_SHOT_SUPPORT_SET)