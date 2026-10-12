"""Optional, process-wide acceleration selected before pandas or sklearn imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AccelerationState:
    requested: str
    backend: str
    cuda_available: bool
    dataframe_accelerated: bool
    cuml_accelerated: bool
    reason: str | None = None


_state = AccelerationState(
    requested="auto",
    backend="cpu",
    cuda_available=False,
    dataframe_accelerated=False,
    cuml_accelerated=False,
    reason="not configured",
)


def configure_acceleration() -> AccelerationState:
    """Enable optional RAPIDS acceleration without making CUDA a requirement."""
    global _state
    requested = os.getenv("ML_ACCELERATION", "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"

    if requested == "cpu":
        _state = AccelerationState(requested, "cpu", False, False, False, "CPU requested")
        return _state

    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("no CUDA device detected")
    except Exception as exc:
        reason = f"CUDA unavailable: {exc}"
        if requested == "cuda":
            logger.warning("%s; using CPU fallback", reason)
        _state = AccelerationState(requested, "cpu", False, False, False, reason)
        return _state

    try:
        import cudf.pandas

        cudf.pandas.install()
    except Exception as exc:
        reason = f"RAPIDS unavailable: {exc}"
        logger.warning("%s; using CPU fallback", reason)
        _state = AccelerationState(requested, "cpu", True, False, False, reason)
        return _state

    cuml_accelerated = False
    if os.getenv("ML_CUML_ACCEL", "false").strip().lower() == "true":
        try:
            from cuml.accel import install

            install()
            cuml_accelerated = True
        except Exception as exc:
            logger.warning("cuML acceleration unavailable: %s", exc)

    _state = AccelerationState(requested, "cuda", True, True, cuml_accelerated)
    logger.info("ML acceleration selected: %s", _state.backend)
    return _state


def get_acceleration_state() -> dict:
    return asdict(_state)


def cuda_available() -> bool:
    return _state.backend == "cuda"
