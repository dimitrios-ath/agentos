"""
Specs are objects that can be added to a Registry,
which is useful for sharing and reproducibility.

Specs are serializable to YAML format and are human readable and manageable.
"""
from typing import Mapping, Union, Any

FlatComponentSpec = Mapping[str, str]
NestedComponentSpec = Mapping[str, Mapping[str, str]]
ComponentSpec = Union[NestedComponentSpec, FlatComponentSpec]
ComponentSpec.identifier_key = "identifier"

# Repo is serialized to a YAML dictionary with the following form:
# {"repo_name": { repo_property_key: repo_property_val}
RepoSpec = Mapping[str, Mapping[str, str]]
RepoSpec.identifier_key = "identifier"

# A paramSet is serialized as a ParameterSetSpec, which is a YAML dictionary
# with the following structure:
# {component_name: {entry_point_name: {param_name: param_val}}
#
# param_value can be any type supported by YAML, which includes:
# scalars (numeric or string), potentially nested lists or dictionaries with
# scalars as leaf values.
#
# Note that you can have a run use complex types via the dependencies
# mechanism which allows a component to depend on other components,
# which themselves can be instances of an arbitrary Python class.
# TODO: Figure out a better type than Any for the leaf type here.
ParameterSetSpec = Mapping[str, Mapping[str, Mapping[str, Any]]]
ParameterSetSpec.identifier_key = "identifier"

RunCommandSpec = Mapping
RunCommandSpec.identifier_key = "identifier"
RunCommandSpec.component_id_key = "component_id"
RunCommandSpec.entry_point_key = "entry_point"
RunCommandSpec.parameter_set_key = "parameter_set"

RunSpec = Mapping
RunSpec.identifier_key = "identifier"

