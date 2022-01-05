import pprint
import mlflow
from typing import Any
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from agentos.registry import Registry
from agentos.parameter_set import ParameterSet
from agentos.exceptions import PythonComponentSystemException
from agentos.specs import RunSpec
from agentos.run_command import RunCommand


class Run:
    """
    Conceptually, a Run represents code execution. More specifically, a Run has
    two distinct uses. First, a Run is used to document an instance of code
    execution and record output associated with it (similar to a logger).
    Second is reproducibility, and for this a Run can optionally hold a
    RunCommand, which, if it exists, can be used to recreate this run, i.e., to
    perform a re-run.

    Structurally, a Run is similar to a logger but provides a bit more
    structure than loggers traditionally do. For example, instead of just a log
    level free text, a Run allows recording of a tag, parameter, and a metric.
    These each have their own semantics and each is represented as a key-value
    pair. Currently, an AgentOS Run is a wrapper around an MLflow Run.

    An MLflow Run is a thin container that holds an RunData and RunInfo object.
    RunInfo contains the run metadata (id, user, timestamp, etc.)
    RunData contains metrics, params, and tags; each of which is a dict.

    AgentOS Run related abstractions are encoded into an MLflowRun as follows:
    - Component Registry incl. root, dependencies, repos -> artifact yaml file
    - Entry point string -> MLflow run tag (MlflowRun.data.tags entry)
    - ParameterSet -> artifact yaml file.
    """

    DEFAULT_EXPERIMENT_ID = "0"
    IS_FROZEN_KEY = "agentos.spec_is_frozen"
    ROOT_COMPONENT_ID_KEY = "agentos.root_component_id"
    ROOT_COMPONENT_REGISTRY_FILENAME = "agentos.components.yaml"
    PARAM_SET_FILENAME = "agentos.parameter_set.yaml"
    ENTRY_POINT_KEY = "agentos.entrypoint"
    # Pass calls to the following functions through to this
    # Run's mlflow_client. All of these take run_id as
    # first arg, and so the pass-through logic also binds
    # self._mlflow_run_id as the first arg of the calls.
    PASS_THROUGH_FN_PREFIXES = [
        "log",
        "set_tag",
        "list_artifacts",
        "download_artifacts",
    ]

    def __init__(
        self,
        run_command: "RunCommand" = None,
        experiment_id: str = None,
        existing_run_id: str = None,
    ) -> None:
        """
        Consider using class factory methods instead of directly using
        __init__. For example Run.from_run_command(), Run.from_existing_run_id.

        :param run_command: Optional RunCommand object.
        :param experiment_id: Optional Experiment ID.
        :param existing_run_id: Optional Run ID.
        """
        assert not (experiment_id and existing_run_id), (
            "existing_run_id cannot be passed with either of run_command or "
            "experiment_id."
        )
        self._run_command = run_command
        self._mlflow_client = MlflowClient()
        if existing_run_id:
            try:
                self._mlflow_client.active_run(existing_run_id)
            except MlflowException as mlflow_exception:
                print(
                    "Error: When creating an AgentOS Run using an "
                    "existing MLflow Run ID, an MLflow run with that ID must "
                    "be available at the default tracking URI, and "
                    f"run_id {existing_run_id} is not."
                )
                raise mlflow_exception
            self._mlflow_run_id = existing_run_id
        else:
            if experiment_id:
                exp_id = experiment_id
            else:
                exp_id = self.DEFAULT_EXPERIMENT_ID
            new_run = self._mlflow_client.create_run(exp_id)
            self._mlflow_run_id = new_run.info.run_id

    def __del__(self):
        self._mlflow_client.set_terminated(self._mlflow_run_id)

    @classmethod
    def from_existing_run_id(cls, run_id: str) -> "Run":
        return cls(existing_run_id=run_id)

    @classmethod
    def from_run_command(
        cls, run_command: RunCommand, experiment_id: str = None
    ) -> "Run":
        return cls(run_command=run_command, experiment_id=experiment_id)

    @staticmethod
    def active_run(caller: Any, fail_if_no_active_run: bool = False) -> "Run":
        """
        A helper function.
        """
        from agentos.component import Component
        if isinstance(caller, Component):
            component = caller
        else:
            try:
                component = caller.__component__
            except AttributeError:
                print(
                    "active_run() was called on an object that is not "
                    "managed by a Component. Specifically, the object passed to "
                    "active_run() must have a ``__component__`` attribute."
                )
        if not component.active_run:
            if fail_if_no_active_run:
                raise PythonComponentSystemException(
                    "active_run() was passed an object managed by a Component "
                    "with no active_run, and fail_if_no_active_run flag was True."
                )
            else:
                run = Run()
                print(
                    "Warning: the object passed to active_run() is managed by a "
                    "Component that has no active_run. Returning a new run "
                    f"(id: {run.identifier}that is not associated with any "
                    "Run object."
                )
            return run
        else:
            return component.active_run.run

    @property
    def _mlflow_run(self):
        return self._mlflow_client.active_run(self._mlflow_run_id)

    @property
    def run_command(self) -> "RunCommand":
        return self._run_command

    @property
    def is_reproducible(self):
        return bool(self.run_command)

    @property
    def identifier(self) -> str:
        return self._mlflow_run.info.run_id

    @property
    def data(self):
        return self._mlflow_run.data

    @property
    def info(self):
        return self._mlflow_run.info

    def __getattr__(self, attr_name):
        prefix_matches = [
            attr_name.startswith(x)
            for x in self.PASS_THROUGH_FN_PREFIXES
        ]
        if any(prefix_matches):
            try:
                from functools import partial
                mlflow_client_fn = getattr(self._mlflow_client, attr_name)
                return partial(mlflow_client_fn, self._mlflow_run_id)
            except AttributeError:
                raise AttributeError(
                    f"No attribute '{attr_name}' could be found in either "
                    f"'{self.__class__} or the MlflowClient object it is "
                    f" wrapping."
                )
        else:
            raise AttributeError(
                f"type object '{self.__class__}' has no attribute "
                f"'{attr_name}'"
            )

    def log_component(self, root_component: "Component") -> None:
        """
        Log a Registry YAML file for the root component being run, and its full
        transitive dependency graph of other components as part of this Run.
        This registry file will contain the component spec and repo spec for
        each component in the root component's dependency graph. Note that a
        Run object contains a component object and thus the root component's
        full dependency graph of other components, and as such does not depend
        on a Registry to provide reproducibility. Like a Component, a Run
        (including its entry point, parameter_set, root component, and the root
        component's full dependency graph) can be dumped into a Registry for
        sharing purposes, which essentially normalizes the Run's root
        component's dependency graph into flat component specs.
        """
        component_dict = root_component.to_registry().to_dict()
        mlflow.log_dict(component_dict, self.ROOT_COMPONENT_REGISTRY_FILENAME)

    def log_parameter_set(self, param_set: ParameterSet) -> None:
        self._mlflow_client.log_dict(
            self._mlflow_run_id,
            param_set.to_spec(),
            self.PARAM_SET_FILENAME
        )

    def print_status(self, detailed: bool = False) -> None:
        if not detailed:
            filtered_tags = {
                k: v
                for k, v in self.tags.items()
                if not k.startswith("mlflow.")
            }
            print(f"\tRun {self.identifier}: {filtered_tags}")
        else:
            pprint.pprint(self.to_spec())

    def to_registry(
        self,
        registry: Registry = None,
        recurse: bool = True,
        force: bool = False,
        include_artifacts: bool = False
    ) -> Registry:
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
            registry.add_run_artifacts(self.identifier,
                                       local_artifact_path)
        if recurse:
            self.run_command.to_registry(
                registry, recurse=recurse, force=force
            )
        return registry

    @property
    def is_publishable(self) -> bool:
        # use like: filtered_tags["is_publishable"] = self.is_publishable
        return self._mlflow_run.data.tags[self.IS_FROZEN_KEY] == "True"

    def to_spec(self) -> RunSpec:
        spec = self._mlflow_run.to_dict()
        run_cmd = self.run_command.to_spec() if self.run_command else None
        spec["run_command"] = run_cmd
        return spec
