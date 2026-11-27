"""Pick mps on Apple Silicon and CPU everywhere else."""


def pick_device(name: str = "auto"):
    import torch

    if name == "auto":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)
