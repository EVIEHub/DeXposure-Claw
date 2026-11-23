from pathlib import Path
from huggingface_hub import hf_hub_download
from torch import nn

import lib
import lib.limix.utils.loading

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tfm(
    tfm_name: str,
    tfm_config: dict,
) -> nn.Module:
    if tfm_name == "LimiX":
        local_ckpt = _REPO_ROOT / "checkpoints" / "LimiX-16M.ckpt"
        if local_ckpt.is_file():
            model_path = str(local_ckpt)
        else:
            model_path = hf_hub_download(
                repo_id="stableai-org/LimiX-16M",
                filename="LimiX-16M.ckpt",
                local_dir=str(_REPO_ROOT / "checkpoints"),
                cache_dir=str(_REPO_ROOT / "checkpoints"),
            )
        return lib.limix.utils.loading.load_model(model_path=model_path)
    else:
        raise ValueError(f"{tfm_name} is not found")
