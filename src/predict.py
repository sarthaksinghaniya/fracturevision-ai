import torch
import torch.nn.functional as F


def predict(model, inputs, device=None):
    """Run model inference and return (pred_class, confidence, probs).

    Args:
        model: PyTorch model that returns logits.
        inputs: Input tensor for a single sample (or batch size 1).
        device: Optional torch.device. If provided, inputs/model are moved there.
    """
    if device is not None:
        model = model.to(device)
        inputs = inputs.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probs_tensor = F.softmax(logits, dim=1)
        probs = probs_tensor[0].detach().cpu().tolist()
        pred_class = int(torch.argmax(probs_tensor, dim=1).item())
        confidence = float(torch.max(probs_tensor, dim=1).values.item())

    return pred_class, confidence, probs
