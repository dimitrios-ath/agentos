from typing import Mapping, Union

FlatComponentSpec = Mapping[str, str]
NestedComponentSpec = Mapping[str, Mapping[str, str]]
ComponentSpec = Union[NestedComponentSpec, FlatComponentSpec]

# Repo is serialized to a dictionary with the following form:
# {"repo_name": { repo_property_key: repo_property_val}
RepoSpec = Mapping[str, Mapping[str, str]]

# A paramSet is serialized as a ParameterSetSpec, which is a dictionary
# with the following structure:
# {component_name: {entry_point_name: {param_name: param_val}}
ParameterSetSpec = Mapping[str, Mapping[str, Mapping[str, str]]]

RunSpec = Mapping
RunSpec.identifier_key = "identifier"
RunSpec.component_id_key = "component_id"
RunSpec.entry_point_key = "entry_point"
RunSpec.parameter_set_key = "parameter_set"
RunSpec.tracker_identifier_key = "parameter_set"
