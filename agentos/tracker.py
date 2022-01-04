import pprint
import mlflow
from typing import Any
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from agentos.parameter_set import ParameterSet
from agentos.exceptions import PythonComponentSystemException


def get_tracker(caller: Any, fail_if_no_active_run: bool = False) -> "Tracker":
    """
    A helper function for developer to to use
    """
    from agentos.component import Component
    if isinstance(caller, Component):
        component = caller
    else:
        try:
            component = caller.__component__
        except AttributeError:
            print(
                "get_tracker() was called on an object that is not "
                "managed by a Component. Specifically, the object passed to "
                "get_tracker() must have a ``__component__`` attribute."
            )
    if not component.active_run:
        if fail_if_no_active_run:
            raise PythonComponentSystemException(
                "get_tracker() was passed an object managed by a Component "
                "with no active_run, and fail_if_no_active_run flag was True."
            )
        else:
            tracker = Tracker()
            print(
                "Warning: the object passed to get_tracker() is managed by a "
                "Component that has no active_run. Returning a new tracker "
                f"(id: {tracker.identifier}that is not associated with any "
                "Run object."
            )
        return tracker
    else:
        return component.active_run.tracker


class Tracker:
    """
    A tracker is an object used to record output from running code.
    Trackers are similar to a logger but provides a bit more structure
    than loggers traditionally do.

    For more on the difference between Run and Tracker, see :py:func:Run:.

    Currently, an AgentOS Tracker is a wrapper around an MLflow Run.

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
    # Tracker's mlflow_client. All of these take run_id as
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
        experiment_id: str = None,
        identifier: str = None
    ) -> None:
        assert not (experiment_id and identifier), (
            "experiment_id and mlflow_run_id cannot both be specified."
        )
        self._mlflow_client = MlflowClient()
        if identifier:
            try:
                self._mlflow_client.get_run(identifier)
            except MlflowException as mlflow_exception:
                print(
                    "Error: When creating an AgentOS Tracker using an "
                    "existing MLflow Run ID, an MLflow run with that ID must "
                    "be available at the default tracking URI, and "
                    f"run_id {identifier} is not."
                )
                raise mlflow_exception
            self._mlflow_run_id = identifier
        else:
            if experiment_id:
                exp_id = experiment_id
            else:
                exp_id = self.DEFAULT_EXPERIMENT_ID
            new_run = self._mlflow_client.create_run(exp_id)
            self._mlflow_run_id = new_run.info.run_id

    def __del__(self):
        self._mlflow_client.set_terminated(self._mlflow_run_id)

    @property
    def _mlflow_run(self):
        return self._mlflow_client.get_run(self._mlflow_run_id)

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

    @property
    def identifier(self) -> str:
        return self._mlflow_run.info.run_id

    @property
    def data(self):
        return self._mlflow_run.data

    @property
    def info(self):
        return self._mlflow_run.info

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