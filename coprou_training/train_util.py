import fnmatch
import torch
import torch.nn.functional as F

def load_pretrained(model, state_dict, pretrained_modules):
    
    # filter according to load_modules (with fnmatch)
    filtered = {
        k: v for k, v in state_dict.items()
        if any(fnmatch.fnmatch(k, pattern) for pattern in pretrained_modules)
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained: {len(filtered)} tensors "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    
def safe_grid_sample(x, grid, **kwargs):
    """
    Run grid_sample safely under AMP (bfloat16/float16).
    Automatically casts to float32 if needed (e.g., for bfloat16).
    """
    if x.dtype == torch.bfloat16 or grid.dtype == torch.bfloat16:
        # bfloat16 → force full precision manually
        with torch.cuda.amp.autocast(enabled=False):
            out = F.grid_sample(x.float(), grid.float(), **kwargs)
        return out.to(x.dtype)
    
    # Otherwise (float16, float32), AMP handles it automatically
    return F.grid_sample(x, grid, **kwargs)
    
def safe_inverse(mat):
    return mat.float().inverse().to(mat.dtype)


 