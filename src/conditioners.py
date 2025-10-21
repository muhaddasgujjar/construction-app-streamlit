import torch

def make_dimension_map(batch_size, H, W, widths, depths, eps=1e-6):
    widths  = widths.view(-1,1,1).clamp(min=eps)
    depths  = depths.view(-1,1,1).clamp(min=eps)
    aspect  = widths / depths
    a = (aspect - 1.0) / 1.0
    m = a.repeat(1, H, W).unsqueeze(1)
    return m.clamp(-1,1)
