import torch


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([
            fn(inputs[i:i + chunk]) 
            for i in range(0, inputs.shape[0], chunk)
        ], dim=0)
    return ret