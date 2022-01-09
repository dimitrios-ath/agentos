import pprint
from pathlib import Path
from typing import Any, TYPE_CHECKING
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow import list_run_infos
from agentos.registry import Registry
from agentos.parameter_set import ParameterSet
from agentos.exceptions import PythonComponentSystemException
from agentos.specs import RunSpec
from agentos.run_command import RunCommand
from agentos.identifiers import RunIdentifier

if TYPE_CHECKING:
    from agentos.component import Component


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
        __init__. For example: Run.from_run_command(),
        Run.from_existing_run_id().

        :param run_command: Optional RunCommand object.
        :param experiment_id: Optional Experiment ID.
        :param existing_run_id: Optional Run ID.
        """
        assert not (experiment_id and existing_run_id), (
            "existing_run_id cannot be passed with either of run_command or "
            "experiment_id."
        )
        self._mlflow_client = MlflowClient()
        self._return_value = None
        self._run_command = None
        if existing_run_id:
            try:
                self._mlflow_client.get_run(existing_run_id)
                # component = self.
                # entry_point =
                # param_set =
                # run_command = RunCommand(component, entry_point, param_set)
                # self.run_command = run_command
            except MlflowException as mlflow_exception:
                print(
                    "Error: When creating an AgentOS Run using an "
                    "existing MLflow Run ID, an MLflow run with that ID must "
                    "be available at the current tracking URI, and "
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

        if run_command:
            self.add_run_command(run_command)

    def __del__(self):
        self._mlflow_client.set_terminated(self._mlflow_run_id)

    @classmethod
    def get_all_runs(cls):
        run_infos = list_run_infos(
            experiment_id=cls.DEFAULT_EXPERIMENT_ID,
            order_by=["attribute.end_time DESC"],
        )
        return [Run.from_existing_run_id(info.run_id) for info in run_infos]

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
                    "managed by a Component. Specifically, the object passed "
                    "to active_run() must have a ``__component__`` attribute."
                )
        if not component.active_run:
            if fail_if_no_active_run:
                raise PythonComponentSystemException(
                    "active_run() was passed an object managed by a Component "
                    "with no active_run, and fail_if_no_active_run flag was "
                    "True."
                )
            else:
                run = Run()
                print(
                    "Warning: the object passed to active_run() is managed by "
                    "a Component that has no active_run. Returning a new run "
                    f"(id: {run.identifier}that is not associated with any "
                    "Run object."
                )
            return run
        else:
            return component.active_run

    @property
    def _mlflow_run(self):
        return self._mlflow_client.get_run(self._mlflow_run_id)

    @property
    def run_command(self) -> "RunCommand":
        return self._run_command

    @property
    def return_value(self) -> str:
        return self._return_value

    @property
    def is_reproducible(self) -> bool:
        return bool(self.run_command)

    @property
    def identifier(self) -> str:
        return self._mlflow_run.info.run_id

    @property
    def data(self) -> dict:
        return self._mlflow_run.data

    @property
    def info(self) -> dict:
        return self._mlflow_run.info

    def __getattr__(self, attr_name):
        prefix_matches = [
            attr_name.startswith(x) for x in self.PASS_THROUGH_FN_PREFIXES
        ]
        if any(prefix_matches):
            try:
                from functools import partial

                mlflow_client_fn = getattr(self._mlflow_client, attr_name)
                return partial(mlflow_client_fn, self._mlflow_run_id)
            except AttributeError as e:
                raise AttributeError(
                    f"No attribute '{attr_name}' could be found in either "
                    f"'{self.__class__} or the MlflowClient object it is "
                    f"wrapping. " + str(e)
                )
        else:
            raise AttributeError(
                f"type object '{self.__class__}' has no attribute "
                f"'{attr_name}'"
            )

    def add_run_command(self, run_command: RunCommand) -> None:
        assert not self._run_command, "run_command already logged."
        self._run_command = run_command
        self.log_component(run_command.component)
        self.log_parameter_set(run_command.parameter_set)
        self.log_entry_point(run_command.entry_point)

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
        self.log_dict(component_dict, self.ROOT_COMPONENT_REGISTRY_FILENAME)

    def log_parameter_set(self, param_set: ParameterSet) -> None:
        self.log_dict(param_set.to_spec(), self.PARAM_SET_FILENAME)

    def log_entry_point(self, entry_point: str) -> None:
        self.set_tag(self.ENTRY_POINT_KEY, str)

    def log_return_value(
        self,
        ret_val: Any,
        format: str = "pickle",
    ):
        """
        Logs the return value of an entry_point run using the specified
        serialization format.

        :param ret_val: The Python object returned by this Run to be logged.
        :param format: Valid values are 'pickle, 'json', or 'yaml'.
        """
        self._return_value = ret_val
        filename_base = self.identifier + "-return_value"
        if format == "pickle":
            import pickle

            filename = filename_base + ".pickle"
            with open(filename, "wb") as f:
                pickle.dump(ret_val, f)
        elif format == "json":
            import json

            filename = filename_base + ".json"
            with open(filename, "w") as f:
                json.dump(ret_val, f)
        elif format == "yaml":
            import yaml

            filename = filename_base + ".yaml"
            with open(filename, "w") as f:
                yaml.dump(ret_val, f)
        else:
            raise PythonComponentSystemException("Invalid format provided")
        self.log_artifact(filename)
        Path(filename).unlink(missing_ok=True)

    def print_status(self, detailed: bool = False) -> None:
        if not detailed:
            filtered_tags = {
                k: v
                for k, v in self.data.tags.items()
                if not k.startswith("mlflow.")
            }
            print(f"\tRun {self.identifier}: {filtered_tags}")
        else:
            pprint.pprint(self.to_spec())

    @classmethod
    def from_registry(
        cls,
        registry: Registry,
        run_id: RunIdentifier,
    ) -> "Run":
        # TODO figure out a way to deserialize an MLflowRun from the registry
        #     and reconcile that with what is in the tracking store.
        # run_spec = registry.get_run_spec(run_id)
        raise NotImplementedError

    @classmethod
    def from_tracking_store(
        cls,
        run_id: RunIdentifier,
    ):
        try:
            run = cls.from_existing_run_id(run_id)
        except MlflowException as e:
            raise MlflowException(
                f"Creating a new MLflowRun (with id {run_id}) "
                "to back this Run object since MLflow was unable to "
                "retrieve the MLflowRun that was used in the Run that "
                f"we are loading from the registry (id: {run_id}). Use the "
                "fail_on_mlflow_run_not_found arg to raise an exception "
                "instead." + e
            )
        return run

    def to_registry(
        self,
        registry: Registry = None,
        recurse: bool = True,
        force: bool = False,
        include_artifacts: bool = False,
    ) -> Registry:
        if not registry:
            from agentos.registry import InMemoryRegistry

            registry = InMemoryRegistry()
        registry.add_run_spec(self.to_spec())
        # If we are writing to a WebRegistry, have local artifacts, and
        # include_artifacts is True, try uploading the artifact files to the
        # registry.
        if (
            include_artifacts
            and hasattr(registry, "add_run_artifacts")
            and self.info.artifact_uri.startswith("file://")
        ):
            local_artifact_path = self.get_artifacts_dir_path()
            registry.add_run_artifacts(self.identifier, local_artifact_path)
        if recurse:
            self.run_command.to_registry(
                registry, recurse=recurse, force=force
            )
        return registry

    @property
    def is_publishable(self) -> bool:
        # use like: filtered_tags["is_publishable"] = self.is_publishable
        try:
            return self._mlflow_run.data.tags[self.IS_FROZEN_KEY] == "True"
        except KeyError:
            return False

    def to_spec(self, flatten: bool = False) -> RunSpec:
        # TODO:
        inner_spec = self._mlflow_run.to_dict()
        run_cmd = self.run_command.to_spec() if self.run_command else None
        inner_spec["run_command"] = run_cmd
        if flatten:
            inner_spec.update({RunSpec.identifier_key: self.identifier})
            return
        else:
            {self.identifier: inner_spec}
