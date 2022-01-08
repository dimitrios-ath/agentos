import sys
import uuid
import importlib
from typing import Union, TypeVar, Dict, Type, Any, Optional, Sequence
from rich import print as rich_print
from rich.tree import Tree
from agentos.run import Run, RunCommand
from agentos.identifiers import ComponentIdentifier
from agentos.specs import ComponentSpec
from agentos.registry import (
    Registry,
    InMemoryRegistry,
)
from agentos.exceptions import RegistryException
from agentos.repo import Repo, InMemoryRepo, GitHubRepo
from agentos.parameter_set import ParameterSet, ParameterSetSpec

# Use Python generics (https://mypy.readthedocs.io/en/stable/generics.html)
T = TypeVar("T")


class Component:
    """
    A Component is a class manager. It provides a standard way for runtime and
    code implementations to communicate about parameters, entry points, and
    dependencies.
    """

    Identifier = ComponentIdentifier

    def __init__(
        self,
        managed_cls: Type[T],
        repo: Repo,
        identifier: "Component.Identifier",
        class_name: str,
        file_path: str,
        dependencies: Dict = None,
        dunder_name: str = None,
    ):
        """
        :param managed_cls: The object this Component manages.
        :param repo: Where the code for this component's managed object is.
        :param identifier: Used to identify the Component.
        :param dependencies: List of other components that self depends on.
        :param dunder_name: Name used for the pointer to this Component on any
                            instances of ``managed_cls`` created by this
                            Component.
        """
        self._managed_cls = managed_cls
        self.repo = repo
        self.identifier = identifier
        self.class_name = class_name
        self.file_path = file_path
        self.dependencies = dependencies if dependencies else {}
        self._dunder_name = dunder_name or "__component__"
        self._requirements = []
        self.active_run = None

    @classmethod
    def from_default_registry(
        cls, name: str, version: str = None
    ) -> "Component":
        return cls.from_registry(Registry.from_default(), name, version)

    @classmethod
    def from_registry(
        cls,
        registry: Registry,
        name: str,
        version: str = None
    ) -> "Component":
        """
        Returns a Component Object from the provided registry, including
        its full dependency tree of other Component Objects.
        If no Registry is provided, use the default registry.
        """
        identifier = Component.Identifier(name, version)
        component_identifiers = [identifier]
        repos = {}
        components = {}
        dependencies = {}
        while component_identifiers:
            component_id = component_identifiers.pop()
            component_spec = registry.get_component_spec_by_id(
                component_id,
                flatten=True
            )
            component_id_from_spec = ComponentIdentifier(
                component_spec["name"], component_spec["version"]
            )
            repo_id = component_spec["repo"]
            if repo_id not in repos.keys():
                repo_spec = registry.get_repo_spec(repo_id)
                repos[repo_id] = Repo.from_spec(
                    repo_spec, registry.base_dir
                )
            component = cls.from_repo(
                repo=repos[repo_id],
                identifier=component_id_from_spec,
                class_name=component_spec["class_name"],
                file_path=component_spec["file_path"],
            )
            components[component_id] = component
            dependencies[component_id] = component_spec.get("dependencies", {})
            for d_id in dependencies[component_id].values():
                component_identifiers.append(
                    Component.Identifier.from_str(d_id)
                )

        # Wire up the dependency graph
        for c_name, component in components.items():
            for attr_name, dependency_name in dependencies[c_name].items():
                dependency = components[dependency_name]
                component.add_dependency(dependency, attribute_name=attr_name)

        return components[identifier]

    @classmethod
    def from_registry_file(
        cls, yaml_file: str, name: str, version: str = None
    ) -> "Component":
        registry = Registry.from_yaml(yaml_file)
        return cls.from_registry(registry, name, version)

    @classmethod
    def from_class(
        cls,
        managed_cls: Type[T],
        name: str = None,
        dunder_name: str = None,
    ) -> "Component":
        name = name if name else managed_cls.__name__
        return cls(
            managed_cls=managed_cls,
            repo=InMemoryRepo(),
            identifier=Component.Identifier(name),
            class_name=managed_cls.__name__,
            file_path=".",
            dunder_name=dunder_name,
        )

    @classmethod
    def from_repo(
        cls,
        repo: Repo,
        identifier: "Component.Identifier",
        class_name: str,
        file_path: str,
        dunder_name: str = None,
    ) -> "Component":
        full_path = repo.get_local_file_path(identifier, file_path)
        assert full_path.is_file(), f"{full_path} does not exist"
        sys.path.append(str(full_path.parent))
        spec = importlib.util.spec_from_file_location(
            f"AOS_MODULE_{class_name.upper()}", str(full_path)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        managed_cls = getattr(module, class_name)
        sys.path.pop()
        return cls(
            managed_cls=managed_cls,
            repo=repo,
            identifier=identifier,
            class_name=class_name,
            file_path=file_path,
            dunder_name=dunder_name,
        )

    @property
    def name(self) -> str:
        return self.identifier.name

    @property
    def version(self):
        return self.identifier.version

    def get_default_entry_point(self):
        try:
            entry_point = self._managed_cls.DEFAULT_ENTRY_POINT
        except AttributeError:
            entry_point = "run"
        return entry_point

    def run(
        self,
        entry_point: str,
        params: Union[ParameterSet, Dict] = None,
        publish_to: Registry = None,
        log_return_value: bool = True,
        return_value_log_format: str = "pickle"
    ) -> Run:
        """
        Run the specified entry point a new instance of this components
        managed object given the specified params, log the results
        and return the Run object.

        :param entry_point: Name of a function to be called on a new
            instance of this component's managed object.
        :param params: A :py:func:agentos.parameter_set.ParameterSet: or
            ParameterSet-like dict containing the entry-point parameters and/or
            parameters to be passed to the __init__() functions of this
            component's dependents during managed object initialization.
        :param publish_to: Optionally, publish the resulting Run object
            to the provided registry.
        :param log_return_value: If the return value of this
        """
        assert not self.active_run, (
            f"Component {self.identifier} already has an active_run, so a "
            "new run is not allowed."
        )
        if params:
            if not isinstance(params, ParameterSet):
                params = ParameterSet(params)
        else:
            params = ParameterSet()
        run_command = RunCommand(self, entry_point, params)
        run = Run.from_run_command(run_command)
        self.active_run = run
        instance = self.get_instance(params=params)
        res = self.call_function_with_param_set(instance, entry_point, params)
        if log_return_value:
            run.log_return_value(res, return_value_log_format)
        self.active_run = None
        if publish_to:
            run.to_registry(publish_to)
        return run

    def call_function_with_param_set(
        self,
        instance: Any,
        function_name: str,
        param_set: ParameterSet
    ) -> Any:
        fn = getattr(instance, function_name)
        assert fn is not None, f"{instance} has no attr {function_name}"
        fn_params = param_set.get_function_params(self.name, function_name)
        print(f"Calling {self.name}.{function_name}(**{fn_params})")
        return fn(**fn_params)

    def add_dependency(
        self, component: "Component", attribute_name: str = None
    ) -> None:
        if type(component) is not type(self):
            raise Exception("add_dependency() must be passed a Component")
        if attribute_name is None:
            attribute_name = component.name
        self.dependencies[attribute_name] = component

    def get_instance(self, params: ParameterSet = None) -> T:
        instantiated = {}
        params = params if params else ParameterSet({})
        return self._get_instance(params, instantiated)

    def _get_instance(self, params: ParameterSet, instantiated: dict) -> T:
        if self.name in instantiated:
            return instantiated[self.name]
        save_init = self._managed_cls.__init__
        self._managed_cls.__init__ = lambda self: None
        instance = self._managed_cls()
        for dep_attr_name, dep_component in self.dependencies.items():
            print(f"Adding {dep_attr_name} to {self.name}")
            dep_instance = dep_component._get_instance(
                params=params, instantiated=instantiated
            )
            setattr(instance, dep_attr_name, dep_instance)
        setattr(instance, self._dunder_name, self)
        self._managed_cls.__init__ = save_init
        self.call_function_with_param_set(instance, "__init__", params)
        instantiated[self.name] = instance
        return instance

    def _handle_repo_spec(self, repos):
        existing_repo = repos.get(self.repo.name)
        if existing_repo:
            if self.repo.to_dict() != existing_repo:
                self.repo.name = str(uuid.uuid4())
        repos[self.repo.name] = self.repo.to_dict()

    def _get_versioned_dependency_dag(
        self, force: bool = False
    ) -> "Component":
        repo_url, version = self.repo.get_version_from_git(
            self.identifier, self.file_path, force
        )
        old_identifier = Component.Identifier(self.identifier.full)
        new_identifier = Component.Identifier(old_identifier.name, version)
        prefixed_file_path = self.repo.get_prefixed_path_from_repo_root(
            new_identifier, self.file_path
        )
        clone = Component(
            managed_cls=self._managed_cls,
            repo=GitHubRepo(identifier=self.repo.identifier, url=repo_url),
            identifier=new_identifier,
            class_name=self.class_name,
            file_path=prefixed_file_path,
            dunder_name=self._dunder_name,
        )
        for attr_name, dependency in self.dependencies.items():
            frozen_dependency = dependency._get_versioned_dependency_dag(
                force=force
            )
            clone.add_dependency(frozen_dependency, attribute_name=attr_name)
        return clone

    def to_spec(self, flatten: bool = False) -> ComponentSpec:
        dependencies = {
            k: str(v.identifier) for k, v in self.dependencies.items()
        }
        component_spec_content = {
            "repo": self.repo.identifier,
            "file_path": str(self.file_path),
            "class_name": self.class_name,
            "dependencies": dependencies,
        }
        if flatten:
            component_spec_content.update(
                {ComponentSpec.identifier_key: str(self.identifier)}
            )
            return component_spec_content
        else:
            return {str(self.identifier): component_spec_content}

    def to_registry(
        self,
        registry: Registry = None,
        recurse: bool = True,
        force: bool = False,
    ) -> Registry:
        """
        Returns a registry containing specs for this component, all of its
        transitive dependents, and the repos of all of them. Throws an
        exception if any of them already exist in the Registry that are
        different unless ``force`` is set to True.

        :param registry: Optionally, add the component spec for this component
                         and each of its transitive dependencies (which are
                         themselves components) to the specified registry.
        :param recurse: If True, check that all transitive dependencies
                        exist in the registry already, and if they don't, then
                        add them. If they do, ensure that they are equal to
                        this component's dependencies (unless ``force`` is
                        specified).
        :param force: Optionally, if a component with the same identifier
                      already exists and is different than the current one,
                      attempt to overwrite the registered one with this one.
        """
        if not registry:
            registry = InMemoryRegistry()
        for c in self.to_dependency_list():
            existing_c_spec = registry.get_component_specs(
                filter_by_name=c.name, filter_by_version=c.version
            )
            if existing_c_spec and not force:
                if existing_c_spec != c.to_spec():
                    raise RegistryException(
                        f"Trying to register a component {c.identifier} that "
                        f"already exists in a different form:\n"
                        f"{existing_c_spec}\n"
                        f"VS\n"
                        f"{c.to_spec()}\n\n"
                        f"To overwrite, specify force=true."
                    )
            registry.add_component_spec(c.to_spec())
            try:
                repo_spec = registry.get_repo_spec(c.repo.identifier)
                if repo_spec != c.repo.to_spec():
                    raise RegistryException(
                        f"A Repo with identifier {c.repo.identifier} already exists"
                        f"in this registry that differs from the one referred "
                        f"to by component {c.identifier}: {repo_spec} vs "
                        f"{c.repo.to_spec()[c.repo.name]}"
                    )
            except LookupError:
                # Repo not yet registered, so so add it to this registry.
                registry.add_repo_spec(c.repo.to_spec())
            if not recurse:
                break
        return registry

    def to_frozen_registry(self, force: bool = False) -> Registry:
        versioned = self._get_versioned_dependency_dag(force)
        return versioned.to_registry()

    def to_dependency_list(
        self, exclude_root: bool = False
    ) -> Sequence["Component"]:
        """
        Return a normalized (i.e. flat) Sequence containing all transitive
        dependencies of this component and (optionally) this component.

        :param exclude_root: Optionally exclude root component from the list.
                             If False, self is first element in list returned.
        :return: a list containing all all of the transitive dependencies
                 of this component (optionally  including the root component).
        """
        component_queue = [self]
        ret_val = set() if exclude_root else set([self])
        while component_queue:
            component = component_queue.pop()
            ret_val.add(component)
            for dependency in component.dependencies.values():
                component_queue.append(dependency)
        return list(ret_val)

    def print_status_tree(self) -> None:
        tree = self.get_status_tree()
        rich_print(tree)

    def get_status_tree(self, parent_tree: Tree = None) -> Tree:
        self_tree = Tree(f"Component: {self.identifier.full}")
        if parent_tree is not None:
            parent_tree.add(self_tree)
        for dep_attr_name, dep_component in self.dependencies.items():
            dep_component.get_status_tree(parent_tree=self_tree)
        return self_tree
