"""Test suite for AgentOS Component."""
import os
import subprocess
from unittest.mock import patch
from unittest.mock import DEFAULT
from agentos import Component


def test_component_repl_demo():
    class SimpleAgent:
        def __init__(self):
            env_name = self.env.__class__.__name__
            print(f"SimpleAgent: AgentOS added self.env: {env_name}")

        def reset_env(self):
            self.env.reset()

    class SimpleEnvironment:
        def reset(self):
            print("SimpleEnvironment.reset() called")

    # Generate Components from Classes
    agent_component = Component.get_from_class(SimpleAgent)
    environment_component = Component.get_from_class(SimpleEnvironment)

    # Add Dependency to SimpleAgent
    agent_component.add_dependency(environment_component, attribute_name="env")

    # Instantiate a SimpleAgent and run reset_env() method
    agent_component.run("reset_env")


def test_component_freezing(tmpdir):
    subprocess.run(["agentos", "init"], cwd=tmpdir, check=True)
    curr_dir = os.getcwd()
    os.chdir(tmpdir)
    try:
        c = Component.get_from_yaml("agent", "agentos.yaml")
        with patch.multiple(
            "agentos.component",
            get_version_from_git=DEFAULT,
            get_prefixed_path_from_repo_root=DEFAULT,
        ) as mocks:
            mocks["get_version_from_git"].return_value = (
                "https://example.com",
                "test_freezing_version",
            )
            mocks[
                "get_prefixed_path_from_repo_root"
            ].return_value = "freeze/test.py"
            c.get_frozen_component_spec()
    finally:
        os.chdir(curr_dir)
