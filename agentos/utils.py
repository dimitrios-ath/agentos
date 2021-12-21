import pprint

import yaml
import mlflow
import shutil
from pathlib import Path
import tempfile
from agentos.specs import ParameterSetSpec

MLFLOW_EXPERIMENT_ID = "0"

AOS_CACHE_DIR = Path.home() / ".agentos_cache"


def log_data_as_yaml_artifact(name: str, data: ParameterSetSpec):
    try:
        tmp_dir_path = Path(tempfile.mkdtemp())
        artifact_path = tmp_dir_path / name
        with open(artifact_path, "w") as file_out:
            file_out.write(yaml.safe_dump(data))
        mlflow.log_artifact(artifact_path)
    finally:
        shutil.rmtree(tmp_dir_path)


def _handle_random_agent(version_string):
    random_path_prefix = Path("example_agents") / Path("random")
    random_rename_map = {
        "agent": f"random_agent=={version_string}",
        "environment": f"random_corridor=={version_string}",
        "policy": f"random_policy=={version_string}",
        "dataset": f"random_dataset=={version_string}",
        "trainer": f"random_trainer=={version_string}",
        "tracker": f"random_tracker=={version_string}",
    }
    return _handle_agent(random_path_prefix, random_rename_map)


def _handle_sb3_agent(version_string):
    sb3_path_prefix = Path("example_agents") / Path("sb3_agent")
    sb3_rename_map = {
        "agent": f"sb3_ppo_agent=={version_string}",
        "environment": f"sb3_cartpole=={version_string}",
        "tracker": f"sb3_tracker=={version_string}",
    }
    return _handle_agent(sb3_path_prefix, sb3_rename_map)


def _handle_acme_r2d2(version_string):
    r2d2_path_prefix = Path("example_agents") / Path("acme_r2d2")
    r2d2_rename_map = {
        "agent": f"acme_r2d2_agent=={version_string}",
        "dataset": f"acme_r2d2_dataset=={version_string}",
        "environment": f"acme_cartpole=={version_string}",
        "network": f"acme_r2d2_network=={version_string}",
        "policy": f"acme_r2d2_policy=={version_string}",
        "tracker": f"acme_tracker=={version_string}",
        "trainer": f"acme_r2d2_trainer=={version_string}",
    }
    return _handle_agent(r2d2_path_prefix, r2d2_rename_map)


def _handle_agent(path_prefix, rename_map):
    aos_root = Path(__file__).parent.parent
    agent_spec = aos_root / path_prefix / "agentos.yaml"
    with open(agent_spec) as file_in:
        registry = yaml.safe_load(file_in)
    registry["repos"] = {}
    registry["repos"]["dev_repo"] = {
        "type": "github",
        "url": "https://github.com/andyk/agentos.git",
    }
    renamed = {}
    for component_name, spec in registry.get("components").items():
        spec["repo"] = "dev_repo"
        spec["file_path"] = str(path_prefix / Path(spec["file_path"]))
        renamed[rename_map[component_name]] = spec
        renamed_dependencies = {}
        for attr_name, dep_name in spec.get("dependencies", {}).items():
            renamed_dependencies[attr_name] = rename_map[dep_name]
        spec["dependencies"] = renamed_dependencies
    registry["components"] = renamed
    registry["latest_refs"] = {
        v.split("==")[0]: v.split("==")[1] for v in rename_map.values()
    }
    return registry


DUMMY_WEB_REGISTRY_DICT = {
    "components": {
        "acme_cartpole==master": {
            "class_name": "CartPole",
            "dependencies": {},
            "file_path": "example_agents/acme_r2d2/../acme_dqn/environment.py",
            "repo": "dev_repo",
        },
        "acme_cartpole==rework_registry": {
            "class_name": "CartPole",
            "dependencies": {},
            "file_path": "example_agents/acme_r2d2/../acme_dqn/environment.py",
            "repo": "dev_repo",
        },
        "acme_r2d2_agent==rework_registry": {
            "class_name": "AcmeR2D2Agent",
            "dependencies": {
                "dataset": "acme_r2d2_dataset==rework_registry",
                "environment": "acme_cartpole==rework_registry",
                "network": "acme_r2d2_network==rework_registry",
                "policy": "acme_r2d2_policy==rework_registry",
                "tracker": "acme_tracker==rework_registry",
                "trainer": "acme_r2d2_trainer==rework_registry",
            },
            "file_path": "example_agents/acme_r2d2/agent.py",
            "repo": "dev_repo",
        },
        "acme_r2d2_dataset==rework_registry": {
            "class_name": "ReverbDataset",
            "dependencies": {
                "environment": "acme_cartpole==rework_registry",
                "network": "acme_r2d2_network==rework_registry",
            },
            "file_path": "example_agents/acme_r2d2/dataset.py",
            "repo": "dev_repo",
        },
        "acme_r2d2_network==rework_registry": {
            "class_name": "R2D2Network",
            "dependencies": {
                "environment": "acme_cartpole==rework_registry",
                "tracker": "acme_tracker==rework_registry",
            },
            "file_path": "example_agents/acme_r2d2/network.py",
            "repo": "dev_repo",
        },
        "acme_r2d2_policy==rework_registry": {
            "class_name": "R2D2Policy",
            "dependencies": {
                "dataset": "acme_r2d2_dataset==rework_registry",
                "environment": "acme_cartpole==rework_registry",
                "network": "acme_r2d2_network==rework_registry",
            },
            "file_path": "example_agents/acme_r2d2/policy.py",
            "repo": "dev_repo",
        },
        "acme_r2d2_trainer==rework_registry": {
            "class_name": "R2D2Trainer",
            "dependencies": {
                "dataset": "acme_r2d2_dataset==rework_registry",
                "environment": "acme_cartpole==rework_registry",
                "network": "acme_r2d2_network==rework_registry",
            },
            "file_path": "example_agents/acme_r2d2/trainer.py",
            "repo": "dev_repo",
        },
        "acme_tracker==rework_registry": {
            "class_name": "AcmeTracker",
            "dependencies": {},
            "file_path": "example_agents/acme_r2d2/../acme_dqn/tracker.py",
            "repo": "dev_repo",
        },
        "random_agent==rework_registry": {
            "class_name": "BasicAgent",
            "dependencies": {
                "dataset": "random_dataset==rework_registry",
                "environment": "random_corridor==rework_registry",
                "policy": "random_policy==rework_registry",
                "tracker": "random_tracker==rework_registry",
                "trainer": "random_trainer==rework_registry",
            },
            "file_path": "example_agents/random/agent.py",
            "repo": "dev_repo",
        },
        "random_corridor==rework_registry": {
            "class_name": "Corridor",
            "dependencies": {},
            "file_path": "example_agents/random/environment.py",
            "repo": "dev_repo",
        },
        "random_dataset==rework_registry": {
            "class_name": "BasicDataset",
            "dependencies": {},
            "file_path": "example_agents/random/dataset.py",
            "repo": "dev_repo",
        },
        "random_policy==rework_registry": {
            "class_name": "RandomPolicy",
            "dependencies": {
                "environment": "random_corridor==rework_registry"
            },
            "file_path": "example_agents/random/policy.py",
            "repo": "dev_repo",
        },
        "random_tracker==rework_registry": {
            "class_name": "BasicTracker",
            "dependencies": {},
            "file_path": "example_agents/random/tracker.py",
            "repo": "dev_repo",
        },
        "random_trainer==rework_registry": {
            "class_name": "BasicTrainer",
            "dependencies": {},
            "file_path": "example_agents/random/trainer.py",
            "repo": "dev_repo",
        },
        "sb3_cartpole==rework_registry": {
            "class_name": "CartPole",
            "dependencies": {},
            "file_path": "example_agents/sb3_agent/environment.py",
            "repo": "dev_repo",
        },
        "sb3_ppo_agent==rework_registry": {
            "class_name": "SB3PPOAgent",
            "dependencies": {
                "environment": "sb3_cartpole==rework_registry",
                "tracker": "sb3_tracker==rework_registry",
            },
            "file_path": "example_agents/sb3_agent/agent.py",
            "repo": "dev_repo",
        },
        "sb3_tracker==rework_registry": {
            "class_name": "SB3Tracker",
            "dependencies": {},
            "file_path": "example_agents/sb3_agent/tracker.py",
            "repo": "dev_repo",
        },
    },
    "latest_refs": {
        "acme_cartpole": "rework_registry",
        "acme_r2d2_agent": "rework_registry",
        "acme_r2d2_dataset": "rework_registry",
        "acme_r2d2_network": "rework_registry",
        "acme_r2d2_policy": "rework_registry",
        "acme_r2d2_trainer": "rework_registry",
        "acme_tracker": "rework_registry",
        "random_agent": "rework_registry",
        "random_corridor": "rework_registry",
        "random_dataset": "rework_registry",
        "random_policy": "rework_registry",
        "random_tracker": "rework_registry",
        "random_trainer": "rework_registry",
        "sb3_cartpole": "rework_registry",
        "sb3_ppo_agent": "rework_registry",
        "sb3_tracker": "rework_registry",
    },
    "repos": {
        "dev_repo": {
            "type": "github",
            "url": "https://github.com/andyk/agentos.git",
        }
    },
}


def generate_dummy_dev_registry():
    registry = {}
    VERSION_STRING = "rework_registry"
    r2d2 = _handle_acme_r2d2(VERSION_STRING)
    _merge_registry_dict(registry, r2d2)
    sb3 = _handle_sb3_agent(VERSION_STRING)
    _merge_registry_dict(registry, sb3)
    rando = _handle_random_agent(VERSION_STRING)
    _merge_registry_dict(registry, rando)
    registry["components"]["acme_cartpole==master"] = {
        "class_name": "CartPole",
        "dependencies": {},
        "file_path": "example_agents/acme_r2d2/../acme_dqn/environment.py",
        "repo": "dev_repo",
    }
    pprint.pprint(registry)
    return registry


def _merge_registry_dict(a, b):
    for key, val in b.items():
        tmp = a.get(key, {})
        tmp.update(val)
        a[key] = tmp
