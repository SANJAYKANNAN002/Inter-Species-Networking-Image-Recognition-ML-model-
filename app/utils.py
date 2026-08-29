# Classes that indicate possible poaching activity
HIGH_RISK_CLASSES = {
    "Hunter",
    "Rifle",
    "Pistol",
    "Knife",
    "X-Bow",
}

# Objects that can provide supporting evidence
MEDIUM_RISK_CLASSES = {
    "Rope",
    "Binocular",
    "Jeep",
}


def calculate_threat(detected_classes):
    """
    Calculate a simple rule-based threat level.

    This is NOT a machine-learning prediction.
    It is a rule-based interpretation of YOLO detections.
    """

    detected = set(detected_classes)

    high_risk_found = detected.intersection(HIGH_RISK_CLASSES)
    medium_risk_found = detected.intersection(MEDIUM_RISK_CLASSES)

    # Multiple high-risk indicators
    if len(high_risk_found) >= 2:
        return "HIGH", (
            "Multiple indicators of possible poaching activity detected."
        )

    # Hunter + weapon
    if "Hunter" in detected and (
        {"Rifle", "Pistol", "Knife", "X-Bow"} & detected
    ):
        return "HIGH", (
            "Hunter and potential weapon detected."
        )

    # Any high-risk class
    if high_risk_found:
        return "HIGH", (
            "Potential poaching-related object/person detected."
        )

    # Supporting objects
    if medium_risk_found:
        return "MEDIUM", (
            "Suspicious supporting equipment detected."
        )

    return "LOW", "No immediate poaching indicators detected."


def get_detected_classes(result):
    """
    Extract class names from a YOLO Results object.
    """

    if result.boxes is None or len(result.boxes) == 0:
        return []

    class_ids = result.boxes.cls.tolist()

    return [
        result.names[int(class_id)]
        for class_id in class_ids
    ]