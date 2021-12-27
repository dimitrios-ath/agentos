import os
import sys
import yaml
import mlflow
import pprint
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, List, TYPE_CHECKING
from contextlib import contextmanager
from mlflow.entities import Run as MLflowRun
from agentos.registry import Registry
from agentos.parameter_set import ParameterSet
from agentos.repo import BadGitStateException, NoLocalPathException
from agentos.specs import RunSpec
from agentos.identifiers import RunIdentifier

# Avoids cicular imports
if TYPE_CHECKING:
    from agentos import Component


class Run:
    MLFLOW_EXPERIMENT_ID = "0"
    PARAM_KEY = "parameter_set.yaml"
    REG_KEY = "components.yaml"
    FROZEN_KEY = "spec_is_frozen"
    ROOT_NAME_KEY = "root_name"
    ENTRY_POINT_KEY = "entry_point"

    def __init__(self, mlflow_run: MLflowRun):
        self._mlflow_run = mlflow_run

    @classmethod
    def from_default_registry(cls, name: str, version: str = None) -> "Run":
        return cls.from_registry(Registry.from_default(), name, version)

    @classmethod
    def from_registry(cls, registry: Registry, run_id: RunIdentifier) -> "Run":
        run_spec = registry.get_run_spec(run_id)
        return cls.from_spec(run_spec)

    @classmethod
    def from_run_spec(cls):
        raise NotImplementedError

    def to_registry(
        self,
        registry: Registry = None,
        recurse: bool = True,
        force: bool = False
    ) -> Registry:
        """
        Returns a registry containing specs for this run and all of its
        transitive dependents, as well the repos of all of them. If recurse
        is True, also adds the component that was run to the registry by
        calling .to_registry() on it, and passing the recurse and force args
        through to that call.

        For details on those flags, see :py:func:agentos.Component.to_registry:
        """
        raise NotImplementedError


def to_dir(self, directory_name: str) -> None:
        """
        Set up a directory with all of the necessary parts for a rerun,
        including a registry file that contains a run_spec for this run,
        and all component_specs and repo specs required for the run.
        Also include copies of source files for all components in same repo as
        the root component of the run.
        """
        from agentos.run_manager import AgentRunManager

        with open("parameter_set.yaml", "w") as param_file:
            param_file.write(yaml.safe_dump(run_spec["parameter_set"]))

        root_response = requests.get(run_spec["root_link"])
        root_data = json.loads(root_response.content)

        rerun_cmd = (
            f'agentos run {root_data["name"]}=={root_data["version"]} '
            f'--entry-point {run_spec["entry_point"]} '
            f"--param-file parameter_set.yaml"
        )
        with open("README.md", "w") as readme_file:
            readme_file.write("## Rerun this agent\n\n```\n")
            readme_file.write(rerun_cmd)
            readme_file.write("\n```")
        spec_url = f"{self.run_api_url}/root_spec"
        spec_response = requests.get(spec_url)
        spec_dict = json.loads(spec_response.content)
        with open("components.yaml", "w") as file_out:
            file_out.write(yaml.safe_dump(spec_dict))
        try:
            tar_url = f"{self.run_api_url}/download_artifact"
            tmp_dir_path = Path(tempfile.mkdtemp())
            requests.get(tar_url)
            tarball_response = requests.get(tar_url)
            tarball_name = "artifacts.tar.gz"
            tarball_path = tmp_dir_path / tarball_name
            with open(tarball_path, "wb") as f:
                f.write(tarball_response.content)
            tar = tarfile.open(tarball_path)
            tar.extractall(path=tmp_dir_path)
            mlflow.start_run(experiment_id=Run.MLFLOW_EXPERIMENT_ID)
            mlflow.set_tag(
                AgentRunManager.RUN_TYPE_TAG, AgentRunManager.LEARN_KEY
            )
            mlflow.log_metric(
                "episode_count",
                run_spec["mlflow_metrics"]["training_episode_count"],
            )
            mlflow.log_metric(
                "step_count", run_spec["mlflow_metrics"]["training_step_count"]
            )
            mlflow.end_run()
            mlflow.start_run(experiment_id=Run.MLFLOW_EXPERIMENT_ID)
            mlflow.set_tag(
                AgentRunManager.RUN_TYPE_TAG, AgentRunManager.RESTORE_KEY
            )
            for file_name in os.listdir(tmp_dir_path):
                if file_name == tarball_name:
                    continue
                file_path = tmp_dir_path / file_name
                mlflow.log_artifact(file_path)
            for name, value in run_spec["mlflow_metrics"].items():
                mlflow.log_metric(name, value)
            mlflow.end_run()
            print("\nRerun agent as follows:")
            print(rerun_cmd)
            print()
        finally:
            shutil.rmtree(tmp_dir_path)

    @staticmethod
    def get_all_runs() -> List["Run"]:
        run_infos = mlflow.list_run_infos(
            experiment_id=Run.MLFLOW_EXPERIMENT_ID,
            order_by=["attribute.end_time DESC"],
        )
        runs = [
            mlflow.get_run(run_id=run_info.run_id) for run_info in run_infos
        ]
        runs = [mlflow.active_run()] + runs
        runs = [run for run in runs if run is not None]
        return [Run(mlflow_run=run) for run in runs]

    @staticmethod
    def get_latest_publishable_run() -> Optional["Run"]:
        all_runs = Run.get_all_runs()
        publishable = [run for run in all_runs if run.is_publishable]
        if len(publishable) == 0:
            return None
        else:
            return publishable[0]

    @staticmethod
    def run_exists(run_id: Optional[str]) -> bool:
        return Run.get_by_id(run_id) is not None

    @staticmethod
    def get_by_id(run_id: Optional[str]) -> Optional["Run"]:
        run_map = {run.id: run for run in Run.get_all_runs()}
        return run_map.get(run_id)

    @staticmethod
    def print_all_status() -> None:
        runs = Run.get_all_runs()
        print("\nRuns:")
        for run in runs:
            run.print_status()
        print()

    @classmethod
    def active_run(cls):
        return cls(mlflow.active_run())

    @staticmethod
    def log_param(name, value):
        mlflow.log_param(name, value)

    @staticmethod
    def log_metric(name, value):
        mlflow.log_metric(name, value)

    @staticmethod
    def set_tag(name, value):
        mlflow.set_tag(name, value)

    @classmethod
    def track(
        cls,
        root_component: "Component",
        fn_name: str,
        params: ParameterSet,
        tracked: bool,
    ):
        if not tracked:
            return cls._untracked_run()
        else:
            return cls._tracked_run(root_component, fn_name, params)

    @classmethod
    @contextmanager
    def _tracked_run(
        cls, root_component: "Component", fn_name: str, params: ParameterSet
    ):
        run = cls(mlflow.start_run(experiment_id=cls.MLFLOW_EXPERIMENT_ID))
        run.log_parameter_set(params)
        run.log_component_spec(root_component)
        run.log_call(root_component.identifier.full, fn_name)
        try:
            yield run
        finally:
            mlflow.end_run()

    @classmethod
    @contextmanager
    def _untracked_run(cls):
        try:
            yield None
        finally:
            pass

    @property
    def id(self) -> str:
        return self._mlflow_run.info.run_id

    @property
    def entry_point(self) -> str:
        return self._mlflow_run.data.params[self.ENTRY_POINT_KEY]

    @property
    def is_publishable(self) -> bool:
        if self.FROZEN_KEY not in self._mlflow_run.data.params:
            return False
        return self._mlflow_run.data.params[self.FROZEN_KEY] == "True"

    @property
    def root_component(self) -> str:
        return self._mlflow_run.data.params[self.ROOT_NAME_KEY]

    @property
    def parameter_set(self) -> Dict:
        return self._get_yaml_artifact(self.PARAM_KEY)

    @property
    def component_spec(self) -> Dict:
        return self._get_yaml_artifact(self.REG_KEY)

    @property
    def tags(self) -> Dict:
        return self.mlflow_data.tags

    @property
    def metrics(self) -> Dict:
        return self.mlflow_data.metrics

    @property
    def params(self) -> Dict:
        return self.mlflow_data.params

    @property
    def mlflow_data(self):
        return self._mlflow_run.data

    @property
    def mlflow_info(self):
        return self._mlflow_run.info

    def _get_yaml_artifact(self, name: str) -> Dict:
        artifacts_dir_path = self.get_artifacts_dir_path()
        artifact_path = artifacts_dir_path / name
        if not artifact_path.is_file():
            return {}
        with artifact_path.open() as file_in:
            return yaml.safe_load(file_in)

    def get_artifacts_dir_path(self) -> Path:
        artifacts_uri = self.mlflow_info.artifact_uri
        if "file://" != artifacts_uri[:7]:
            raise Exception(f"Non-local artifacts path: {artifacts_uri}")
        slice_count = 7
        if sys.platform in ["win32", "cygwin"]:
            slice_count = 8
        return Path(artifacts_uri[slice_count:]).absolute()

    def print_status(self, detailed: bool = False) -> None:
        if not detailed:
            filtered_tags = {
                k: v
                for k, v in self.tags.items()
                if not k.startswith("mlflow.")
            }
            filtered_tags["is_publishable"] = self.is_publishable
            print(f"\tRun {self.id}: {filtered_tags}")
        else:
            pprint.pprint(self.to_spec())

    def to_dict(self) -> RunSpec:
        artifact_paths = [str(p) for p in self._get_artifact_paths()]
        mlflow_info_dict = {
            "artifact_uri": self.mlflow_info.artifact_uri,
            "end_time": self.mlflow_info.end_time,
            "experiment_id": self.mlflow_info.experiment_id,
            "lifecycle_stage": self.mlflow_info.lifecycle_stage,
            "run_id": self.mlflow_info.run_id,
            "run_uuid": self.mlflow_info.run_uuid,
            "start_time": self.mlflow_info.start_time,
            "status": self.mlflow_info.status,
            "user_id": self.mlflow_info.user_id,
        }
        return {
            "id": self.id,
            "is_publishable": self.is_publishable,
            "root_component": self.root_component,
            "entry_point": self.entry_point,
            "parameter_set": self.parameter_set,
            "component_spec": self.component_spec,
            "artifacts": artifact_paths,
            "mlflow_info": mlflow_info_dict,
            "mlflow_data": self.mlflow_data.to_dictionary(),
        }

    def _get_artifact_paths(self) -> List[Path]:
        artifacts_dir = self.get_artifacts_dir_path()
        artifact_paths = []
        skipped_artifacts = [
            self.PARAM_KEY,
            self.REG_KEY,
        ]
        for name in os.listdir(self.get_artifacts_dir_path()):
            if name in skipped_artifacts:
                continue
            artifact_paths.append(artifacts_dir / name)

        exist = [p.exists() for p in artifact_paths]
        assert all(exist), f"Missing artifact paths: {artifact_paths}, {exist}"
        return artifact_paths

    def publish(self) -> None:
        run_id = self.to_registry(default_registry)
        print(f"Published Run {run_id} to {default_registry}.")

    def to_registry(self, registry: Registry) -> str:
        if not self.is_publishable:
            raise Exception("Run not publishable; Spec is not frozen!")
        result = registry.add_run_spec(self.to_spec())
        run_id = result["id"]
        registry.add_run_artifacts(run_id, self._get_artifact_paths())
        return run_id

    def log_parameter_set(self, params: ParameterSet) -> None:
        self.log_data_as_yaml_artifact(self.PARAM_KEY, params.to_spec())

    def log_component_spec(self, root_component: "Component") -> None:
        frozen = None
        try:
            frozen = root_component.to_frozen_registry()
            # FIXME - Will need to be adapted for WebRegistry
            self.log_data_as_yaml_artifact(self.REG_KEY, frozen.to_dict())
        except (BadGitStateException, NoLocalPathException) as exc:
            print(f"Warning: component is not publishable: {str(exc)}")
            unfrozen = root_component.to_registry()
            # FIXME - Will need to be adapted for WebRegistry
            self.log_data_as_yaml_artifact(self.REG_KEY, unfrozen.to_dict())
        mlflow.log_param(self.FROZEN_KEY, frozen is not None)

    def log_call(self, root_name: str, fn_name: str) -> None:
        mlflow.log_param(self.ROOT_NAME_KEY, root_name)
        mlflow.log_param(self.ENTRY_POINT_KEY, fn_name)

    def log_data_as_yaml_artifact(self, name: str, data: dict):
        try:
            tmp_dir_path = Path(tempfile.mkdtemp())
            artifact_path = tmp_dir_path / name
            with open(artifact_path, "w") as file_out:
                file_out.write(yaml.safe_dump(data))
            mlflow.log_artifact(artifact_path)
        finally:
            shutil.rmtree(tmp_dir_path)
