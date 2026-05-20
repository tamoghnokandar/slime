import logging
import os
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def enabled():
    return os.getenv("SLIME_STARTUP_PROFILE") == "1"


@contextmanager
def startup_phase(name):
    if not enabled():
        yield
        return
    nvtx = None
    if os.getenv("SLIME_STARTUP_NVTX") == "1":
        try:
            import torch

            nvtx = torch.cuda.nvtx
            nvtx.range_push(name)
        except Exception:
            nvtx = None
    start = time.perf_counter()
    logger.info("SLIME_STARTUP_PROFILE phase=%s event=start", name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("SLIME_STARTUP_PROFILE phase=%s event=end elapsed_s=%.3f", name, elapsed)
        if nvtx is not None:
            nvtx.range_pop()
