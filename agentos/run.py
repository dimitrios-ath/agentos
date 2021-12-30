import os
import sys
import yaml
import mlflow
import pprint
import shutil
import tempfile
import tarfile
from urllib.parse import urlparse
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
    from agentos.component import Component

"""
An MLflow Run is a thin container that holds an RunData and RunInfo object.
RunInfo contains the run metadata (id, user, timestamp, etc.)
RunData contains metrics, params, and tags. each is a dict

In MLflow, the concept of a EntryPoint exists in the context of a Project run.
A Project Run is a wrapper around an MLflow tracking Run. A project Run
uses Run Tags to log all sorts of metadata, including the entry point per
https://github.com/mlflow/mlflow/blob/v1.22.0/mlflow/projects/utils.py#L225
and 
https://github.com/mlflow/mlflow/blob/v1.22.0/mlflow/utils/mlflow_tags.py
"""


class Run:
    MLFLOW_EXPERIMENT_ID = "0"
    IS_FROZEN_KEY = "agentos.spec_is_frozen"
    ROOT_COMPONENT_ID_KEY = "agentos.root_component_id"
    ROOT_COMPONENT_REGISTRY_FILENAME = "agentos.root_component_registry"
    PARAM_SET_KEY = "agentos.parameter_set"
    ENTRY_POINT = "agentos.entrypoint"

    def __init__(self, mlflow_run: MLflowRun) -> None:
        self._mlflow_run = mlflow_run

    @classmethod
    def from_default_registry(cls, run_id: RunIdentifier) -> "Run":
        return cls.from_registry(Registry.from_default(), run_id)

    @classmethod
    def from_registry(cls, registry: Registry, run_id: RunIdentifier) -> "Run":
        run_spec = registry.get_run_spec(run_id)
        return cls.from_spec(run_spec)

    @classmethod
    def from_spec(cls, run_spec: RunSpec):
        mlflow_run = MLflowRun(
            run_info=run_spec["info"], run_data=run_spec["data"]
        )
        return cls(mlflow_run)

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
        run.log_component_registry(root_component)
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
        return self._mlflow_run.data.tags[self.ENTRY_POINT_KEY]

    @property
    def is_publishable(self) -> bool:
        if self.IS_FROZEN_KEY not in self._mlflow_run.data.params:
            return False
        return self._mlflow_run.data.tags[self.IS_FROZEN_KEY] == "True"

    @property
    def root_component_identifier(self) -> str:
        return self._mlflow_run.data.tags[self.ROOT_COMPONENT_ID_KEY]

    @property
    def root_component_spec(self) -> Dict:
        return self._get_yaml_artifact(self.ROOT_COMPONENT_REGISTRY_FILENAME)

    @property
    def info(self):
        return self._mlflow_run.info

    @property
    def data(self):
        return self._mlflow_run.data

    @property
    def tags(self) -> Dict:
        return self.data.tags

    @property
    def metrics(self) -> Dict:
        return self.data.metrics

    @property
    def params(self) -> Dict:
        return self.parameter_set

    @property
    def parameter_set(self) -> Dict:
        return self._mlflow_run.data.params

    def publish(self) -> None:
        """
        This function is like :py:func:to_registry: but it writes the Run to
        the default registry, whereas :py:func:to_registry: writes the Run
        either to an explicitly provided registry object, or to a new
        InMemoryRegistry.
        """
        if not self.is_publishable:
            raise Exception("Run not publishable; Spec is not frozen!")
        default_registry = Registry.from_default()
        run_id = self.to_registry(default_registry)
        print(f"Published Run {run_id} to {default_registry}.")

    def to_registry(
        self,
        registry: Registry = None,
        recurse: bool = True,
        force: bool = False,
        include_artifacts = False
    ) -> Registry:
        """
        Returns a registry a run spec for this run. If recurse
        is True, also adds the component that was run to the registry by
        calling .to_registry() on it, and passing the given registry arg as
        well as the recurse and force args through to that call.

        For details on those flags, see :py:func:agentos.Component.to_registry:
        """
        if not registry:
            from agentos.registry import InMemoryRegistry
            registry = InMemoryRegistry()
        registry.add_run_spec(self.to_spec())
        # If we are writing to a WebRegistry and have local, then optionally
        # (per function arg) try uploading the artifact files to the registry.
        if (
            include_artifacts and
            hasattr(registry, "add_run_artifacts") and
            self.info.artifact_uri.startswith("file://")
        ):
            local_artifact_path = self.get_artifacts_dir_path()
            registry.add_run_artifacts(self.id, local_artifact_path)
        if recurse:
            registry.add_component(self.root_component_spec, recurse, force)
        return registry

    def rerun(self) -> "Run":
        """
        Create a new run using the same root component, entry point, and
        params as this Run.

        :return: a new Run object representing the rerun.
        """
        root_component = Component.from_spec(self.root_component_spec)
        return root_component.run(self.entry_point, self.parameter_set)

    def to_dir(self, dir_name: str) -> None:
        """
        Set up a directory with all of the necessary parts for a rerun,
        including a registry file that contains a run_spec for this run,
        and all component_specs and repo specs required for the run.
        Also include copies of source files for all components in same repo as
        the root component of the run.
        """
        from agentos.run_manager import AgentRunManager

        # Make param file.
        with open(Path(dir_name) / "parameter_set.yaml", "w") as param_file:
            param_file.write(yaml.safe_dump(self.to_spec()["parameter_set"]))
        rerun_cmd = (
            f'agentos run {self.root_component.identifier} '
            f'--entry-point {self.entry_point} '
            f"--param-file parameter_set.yaml"
        )
        with open("README.md", "w") as readme_file:
            readme_file.write("## Rerun this agent\n\n```\n")
            readme_file.write(rerun_cmd)
            readme_file.write("\n```")

        # Make Registry (components.yaml) file.
        self.root_component.to_registry(Path(dir_name) / "components.yaml")

        # Make artifacts dir (if any).
        try:
            tmp_dir_path = Path(tempfile.mkdtemp())
            tarball_name = "artifacts.tar.gz"
            tarball_path = self.get_artifacts_dir_path() / tarball_name
            tar = tarfile.open(tarball_path)
            tar.extractall(path=tmp_dir_path)

            # Log new MLflow runs for this new dir.
            mlflow.start_run(experiment_id=Run.MLFLOW_EXPERIMENT_ID)
            mlflow.set_tag(
                AgentRunManager.RUN_TYPE_TAG, AgentRunManager.LEARN_KEY
            )
            mlflow.log_metric(
                "episode_count",
                self.metrics["training_episode_count"],
            )
            mlflow.log_metric(
                "step_count", self.metrics["training_step_count"]
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
            for name, value in self.metrics.items():
                mlflow.log_metric(name, value)
            mlflow.end_run()
            print("\nRerun agent as follows:")
            print(rerun_cmd)
            print()
        finally:
            shutil.rmtree(tmp_dir_path)

    def _get_yaml_artifact(self, name: str) -> Dict:
        artifacts_dir_path = self.get_artifacts_dir_path()
        artifact_path = artifacts_dir_path / name
        if not artifact_path.is_file():
            return {}
        with artifact_path.open() as file_in:
            return yaml.safe_load(file_in)

    def get_artifacts_dir_path(self) -> Path:
        artifacts_uri = self.info.artifact_uri
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

    def to_spec(self) -> RunSpec:
        return self._mlflow_run.to_dictionary()

    def log_parameter_set(self, params: ParameterSet) -> None:
        mlflow.log_parameters(params.to_spec())

    def log_component_registry(self, root_component: "Component") -> None:
        """
        Log a Registry for the root component being run, and its full
        transitive dependency graph of other components as part of this Run.
        This registry will contain the component spec and repo spec for each
        component in the root component's dependency graph.
        """
        frozen = None
        try:
            root_id = root_component.identifier
            frozen_reg = root_component.to_frozen_registry()
            frozen = frozen_reg.get_component_spec_by_id(root_id)
            mlflow.log_dict(self.ROOT_COMPONENT_REGISTRY_FILENAME, frozen)
        except (BadGitStateException, NoLocalPathException) as exc:
            print(
                f"Warning: Generating frozen component registry for {root_id} "
                f"failed while logging it to to run {self.id}. Logging "
                f"unfrozen component registry run instead.\n{str(exc)}"
            )
            unfrozen = root_component.to_registry().to_dict()
            mlflow.log_dict(self.ROOT_COMPONENT_REGISTRY_FILENAME, unfrozen)
        mlflow.log_tag(self.IS_FROZEN_KEY, frozen is not None)

    def log_call(self, root_name: str, fn_name: str) -> None:
        mlflow.log_param(self.ROOT_COMPONENT_ID_KEY, root_name)
        mlflow.log_param(self.ENTRY_POINT_KEY, fn_name)
