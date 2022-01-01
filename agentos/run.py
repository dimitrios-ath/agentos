import mlflow
import pprint
from typing import TYPE_CHECKING
from mlflow.tracking.client import MlflowClient
from mlflow.exceptions import MlflowException
from agentos.registry import Registry
from agentos.parameter_set import ParameterSet
from agentos.repo import BadGitStateException, NoLocalPathException
from agentos.specs import RunSpec
from agentos.identifiers import RunIdentifier

# Avoids cicular imports
if TYPE_CHECKING:
    from agentos.component import Component


class Run:
    """
    A Run keeps track of the inputs involved for running a Component
    Entry Point. It does not directly worry about the outputs. That is, it
    does not provide a logging-like interface for tracking things
    like metrics or artifacts. Instead it owns a Tracker object, which
    is where the logging-like functionality lives.

    You can think of a Run as a glorified normalized dictionary containing
    the pointers to params, and versions of code necessary to reproduce
    the setting up running of a component. Whereas a Run's Tracker is more like
    a client to a backing store used for logging things.

    Our concept of a Run is inspired by the MLflow Project Run abstraction.
    In MLflow runs of Projects (which are roughly analogous to our components)
    are intertwined with MLflow's concept of Runs for tracking purposes. In
    MLflow, a Project Run is a wrapper around an MLflow tracking Run.

    In MLflow, an entry point exists in the context of a Project Run. A project
    Run uses Tags on the underlying tracking run to log all sorts of metadata,
    including the entry point, per
    https://github.com/mlflow/mlflow/blob/v1.22.0/mlflow/projects/utils.py#L225
    and
    https://github.com/mlflow/mlflow/blob/v1.22.0/mlflow/utils/mlflow_tags.py
    """

    def __init__(
        self,
        component: "Component",
        entry_point: str,
        parameter_set: ParameterSet,
        experiment_id: str = None,
        tracker: "Tracker" = None,
    ):
        assert not (tracker and experiment_id), (
            "tracker and experiment_id cannot both be passed at the same time."
        )
        self._component = component
        self._entry_point = entry_point
        self._parameter_set = parameter_set
        self._experiment_id = experiment_id

        if tracker:
            self._tracker = tracker
        else:
            self._tracker = Tracker(experiment_id=experiment_id)
            self._tracker.set_tag(RunSpec.entry_point_key)
            self._tracker.log_parameter_set(parameter_set)
            try:
                component = component.to_frozen_registry(component.identifier)
            except (BadGitStateException, NoLocalPathException) as exc:
                print(
                    "Warning: Generating frozen component registry for "
                    f"{component.identifier} failed while logging it to "
                    f"run {self.id}. Saving unfrozen component registry "
                    f"to run instead.\n{str(exc)}"
                )
            self._tracker.log_component(component)

    @property
    def identifier(self):
        return self.tracker.identifier

    @property
    def tracker(self):
        return self._tracker

    @classmethod
    def from_default_registry(cls, run_id: RunIdentifier) -> "Run":
        return cls.from_registry(Registry.from_default(), run_id)

    @classmethod
    def from_registry(
        cls,
        registry: Registry,
        run_id: RunIdentifier,
        fail_on_mlflow_run_not_found: bool = False
    ) -> "Run":
        run_spec = registry.get_run_spec(run_id)
        component = Component.from_registry(
            registry, run_spec[RunSpec.component_id_key]
        )
        try:
            tracker = Tracker(mlflow_run_id=run_spec[RunSpec.identifier_key])
        except MlflowException as e:
            if fail_on_mlflow_run_not_found:
                raise e
            tracker = Tracker()
            print(
                f"Creating a new MLflowRun (with id {tracker.identifier}) to "
                "back this Run object since MLflow was unable to retrieve "
                "the MLflowRun that was used in the Run that we are loading "
                f"from the registry (id: {run_spec[RunSpec.identifier_key]}). "
                "Use the fail_on_mlflow_run_not_found arg to raise an "
                "exception instead."
            )
        return cls(
            component=component,
            entry_point=run_spec[RunSpec.entry_point_key],
            parameter_set=run_spec[RunSpec.parameter_set_key],
            tracker=tracker
        )

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
        include_artifacts: bool = False
    ) -> Registry:
        """
        Returns a registry (which may optionally already exist) containing a
        run spec for this run. If recurse is True, also adds the component that
        was run to the registry by calling ``.to_registry()`` on it, and
        passing the given registry arg as well as the recurse and force args
        through to that call.

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
            registry.add_run_artifacts(self.identifier, local_artifact_path)
        if recurse:
            registry.add_component(self._component, recurse, force)
        return registry

    def rerun(self) -> "Run":
        """
        Create a new run using the same root component, entry point, and
        params as this Run.

        :return: a new Run object representing the rerun.
        """
        root_component = Component.from_spec(self.root_component_spec)
        return root_component.run(self.entry_point, self.parameter_set)

    def to_spec(self) -> RunSpec:
        return {
           RunSpec.identifier_key: self.identifier,
           RunSpec.component_id_key: self._component.identifier,
           RunSpec.entry_point_key: self._entry_point,
           RunSpec.parameter_set_key: self._parameter_set.to_spec()
        }

    @property
    def is_publishable(self) -> bool:
        # use like: filtered_tags["is_publishable"] = self.is_publishable
        return self._mlflow_run.data.tags[self.IS_FROZEN_KEY] == "True"


class Tracker:
    """
    An AgentOS Tracker is a wrapper around an MLflow Run.

    An MLflow Run is a thin container that holds an RunData and RunInfo object.
    RunInfo contains the run metadata (id, user, timestamp, etc.)
    RunData contains metrics, params, and tags. each is a dict

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

    def __init__(
        self,
        experiment_id: str = None,
        mlflow_run_id: str = None
    ) -> None:
        assert not (experiment_id and mlflow_run_id), (
            "experiment_id and mlflow_run_id cannot both be specified."
        )
        self._mlflow_client = MlflowClient()
        if mlflow_run_id:
            try:
                self._mlflow_client.get_run(mlflow_run_id)
            except MlflowException as mlflow_exception:
                print(
                    "Error: When creating an AgentOS Tracker from an MLflow "
                    "Run, the MLflow run must be available at the default "
                    f"tracking URI, and run_id {mlflow_run_id} is not."
                )
                raise mlflow_exception
            self._mlflow_run_id = mlflow_run_id
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
        try:
            return getattr(self._mlflow_run, attr_name)
        except AttributeError:
            raise AttributeError(
                f"No attribute '{attr_name}' could be found in either "
                f"'{self.__class__} or the MLflowRun object it is wrapping."
            )

    @property
    def identifier(self) -> str:
        return self._mlflow_run.info.run_id

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