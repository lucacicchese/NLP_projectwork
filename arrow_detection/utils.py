import logging

import torch


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_best_cuda_device() -> str:
    """
    Return the CUDA device with the most free memory (e.g. 'cuda:1').
    Falls back to CPU when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return "cpu"

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            logger.info(
                f"GPU {idx}: free={free_bytes / (1024**3):.2f} GiB / total={total_bytes / (1024**3):.2f} GiB"
            )
            if free_bytes > best_free:
                best_free = free_bytes
                best_idx = idx
        except Exception as e:
            logger.warning(f"Could not read memory info for GPU {idx}: {e}")

    return f"cuda:{best_idx}"
