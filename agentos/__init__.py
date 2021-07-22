"""The ``agentos`` module provides an API for building learning agents."""

from agentos.version import VERSION as __version__  # noqa: F401
from agentos.core import (
    Agent,
    Policy,
    Environment,
    run_agent,
    rollout,
    rollouts,
    save_data,
    save_tensorflow,
    restore_data,
    restore_tensorflow,
    saved_data,  # TODO - REMOVE ME?
)

__all__ = [
    "Agent",
    "Policy",
    "Environment",
    "run_agent",
    "rollout",
    "rollouts",
    "save_data",
    "save_tensorflow",
    "restore_data",
    "restore_tensorflow",
    "saved_data",
]
