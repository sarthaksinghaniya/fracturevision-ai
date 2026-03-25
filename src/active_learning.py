def should_ask_feedback(confidence, threshold=0.75):
    """Return True when model confidence is below the feedback threshold."""
    return float(confidence) < float(threshold)
