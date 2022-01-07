import copy
import yaml
from immutables import Map
from typing import TypeVar, Dict
from agentos.specs import ParameterSetSpec

# Use Python generics (https://mypy.readthedocs.io/en/stable/generics.html)
T = TypeVar("T")


class ParameterSet:
    """
    This object is used to encapsulate a set of parameters that are used to
    initialize a Component dependency DAG and to run methods on this DAG.
    """

    def __init__(self, parameters: ParameterSetSpec = None):
        self._parameters = Map(parameters) if parameters else Map

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return True
        else:
            return self is other

    def __hash__(self):
        return hash(self._parameters)

    @classmethod
    def from_yaml(cls, file_path) -> "ParameterSet":
        parameters = {}
        if file_path is not None:
            with open(file_path) as file_in:
                parameters = yaml.safe_load(file_in)
        return ParameterSet(parameters=parameters)

    def update(self, component_name: str, fn_name: str, params: Dict) -> None:
        component_params = self._parameters.get(component_name, {})
        fn_params = component_params.get(fn_name, {})
        fn_params.update(params)
        component_params[fn_name] = fn_params
        self._parameters = self._parameters.set(
            component_name, component_params
        )

    def get_component_params(self, component_name: str) -> Dict:
        return self._parameters.get(component_name, {})

    def get_function_params(
        self, component_name: str, function_name: str
    ) -> Dict:
        component_params = self.get_component_params(component_name)
        fn_params = component_params.get(function_name, {})
        return fn_params if fn_params else {}

    def get_param(
        self, component_name: str, function_name: str, param_key: str
    ) -> Dict:
        fn_params = self.get_function_params(component_name, function_name)
        param = fn_params.get(param_key, {})
        return param if param else {}

    def to_spec(self) -> ParameterSetSpec:
        return copy.deepcopy(self._parameters)
