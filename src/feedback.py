import json
from pathlib import Path


def save_feedback(image_path, predicted_label, correct_label, confidence):
    """Append a feedback record to feedback/feedback_data.json."""
    feedback_dir = Path("feedback")
    feedback_file = feedback_dir / "feedback_data.json"
    feedback_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    if feedback_file.exists():
        with feedback_file.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, list):
                entries = loaded

    entry = {
        "image": str(image_path),
        "predicted": str(predicted_label),
        "correct": str(correct_label),
        "confidence": float(confidence),
    }
    entries.append(entry)

    with feedback_file.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, ensure_ascii=False)

    return entry
