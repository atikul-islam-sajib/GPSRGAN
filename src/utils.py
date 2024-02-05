def total_trainable_params(model):
    """
    Calculate the total number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params
