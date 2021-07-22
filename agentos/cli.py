"""AgentOS command line interface (CLI).

The CLI allows creation of a simple template agent.
"""
import agentos
import shutil
import subprocess
import os
import uuid
import sys
import yaml
import click
from datetime import datetime
import importlib.util
from pathlib import Path
import configparser
import importlib
import statistics


CONDA_ENV_FILE = Path("./templates/conda_env.yaml")
MLFLOW_PROJECT_FILE = Path("./templates/MLProject")
AGENT_DEF_FILE = Path("./templates/agent.py")
ENV_DEF_FILE = Path("./templates/environment.py")
POLICY_DEF_FILE = Path("./templates/policy.py")
AGENT_INI_FILE = Path("./templates/agent.ini")


def get_data_location(package_location):
    return Path(package_location).absolute() / "data"


def get_backups_location(package_location):
    return Path(package_location).absolute() / "backups"


INIT_FILES = [
    CONDA_ENV_FILE,
    MLFLOW_PROJECT_FILE,
    AGENT_DEF_FILE,
    ENV_DEF_FILE,
    POLICY_DEF_FILE,
    AGENT_INI_FILE,
]


@click.group()
@click.version_option()
def agentos_cmd():
    pass


@agentos_cmd.command()
@click.argument("package_name", metavar="PACKAGE_NAME")
@click.option(
    "--package-location",
    "-l",
    metavar="PACKAGE_LOCATION",
    type=click.Path(),
    default="./.acr",
    help="Path to AgentOS Component Registry installation directory",
)
@click.option(
    "--agent-file",
    "-f",
    type=click.Path(exists=True),
    default="./agent.ini",
    help="Path to agent definition file (agent.ini).",
)
def install(package_name, package_location, agent_file):
    """Installs PACKAGE_NAME"""
    package_location = Path(package_location).absolute()
    registry_entry = get_registry_entry(package_name)
    if confirm_package_installation(registry_entry, package_location):
        create_package_directory_structure(package_location)
        release_entry = get_release_entry(registry_entry)
        repo = clone_package_repo(release_entry, package_location)
        checkout_release_hash(release_entry, repo)
        update_agent_ini(registry_entry, release_entry, repo, agent_file)
        install_requirements(repo, release_entry)
    else:
        raise Exception("Aborting installation...")


def get_registry_entry(package_name):
    agentos_root_path = Path(__file__).parent.parent
    registry_path = agentos_root_path / "registry.yaml"
    if not registry_path.is_file():
        raise Exception(f"Could not find AgentOS registry at {registry_path}")
    with open(registry_path) as file_in:
        registry = yaml.full_load(file_in)
    if package_name not in registry:
        raise click.BadParameter(f'Cannot find package "{package_name}"')
    registry[package_name]["_name"] = package_name
    return registry[package_name]


def confirm_package_installation(registry_entry, location):
    answer = input(
        f'ACR will install component {registry_entry["_name"]} '
        f"to {location}.  Continue? (Y/N) "
    )
    return answer.strip().lower() == "y"


def create_package_directory_structure(package_location):
    os.makedirs(package_location, exist_ok=True)


def get_release_entry(registry_entry):
    # TODO - allow specification of release
    return registry_entry["releases"][0]


def clone_package_repo(release, location):
    repo_name = release["github_url"].split("/")[-1]
    clone_destination = (Path(location) / repo_name).absolute()
    if clone_destination.exists():
        raise click.BadParameter(f"{clone_destination} already exists!")
    cmd = ["git", "clone", release["github_url"], clone_destination]
    result = subprocess.run(cmd)
    assert result.returncode == 0, "Git returned non-zero on repo checkout"
    assert clone_destination.exists(), f"Unable to clone repo {repo_name}"
    return clone_destination


def checkout_release_hash(release, repo):
    curr_dir = os.getcwd()
    os.chdir(repo)
    git_hash = release["hash"]
    cmd = ["git", "checkout", "-q", git_hash]
    result = subprocess.run(cmd)
    assert result.returncode == 0, f"FAILED: checkout {git_hash} in {repo}"
    os.chdir(curr_dir)


def update_agent_ini(registry_entry, release_entry, repo, agent_file):
    print(repo)
    config = configparser.ConfigParser()
    config.read(agent_file)
    if registry_entry["type"] == "environment":
        section = "Environment"
    elif registry_entry["type"] == "policy":
        section = "Policy"
    else:
        raise Exception(f"Component component type: {registry_entry['type']}")

    # TODO - allow multiple components of same type installed
    if section in config:
        print(
            f"Replacing current environment {dict(config[section])} "
            f'with {registry_entry["_name"]}'
        )
    module_path = Path(repo).absolute()
    file_path = (module_path / release_entry["file_path"]).absolute()
    config[section]["file_path"] = str(file_path)
    config[section]["class_name"] = release_entry["class_name"]
    config[section]["python_path"] = str(module_path)
    with open(agent_file, "w") as out_file:
        config.write(out_file)


# TODO - automatically install?
def install_requirements(repo, release_entry):
    req_path = (repo / release_entry["requirements_path"]).absolute()
    print("\nInstall component requirements with the following command:")
    print(f"\n\tpip install -r {req_path}\n")


def validate_agent_name(ctx, param, value):
    if " " in value or ":" in value or "/" in value:
        raise click.BadParameter("name may not contain ' ', ':', or '/'.")
    return value


def create_agent_directory_structure(package_location):
    os.makedirs(package_location, exist_ok=True)
    os.makedirs(get_data_location(package_location), exist_ok=True)
    os.makedirs(get_backups_location(package_location), exist_ok=True)


def create_core_data(package_location):
    decorate_save_data_fns(package_location)
    agentos.save_data("step_count", 0)
    agentos.save_data("episode_count", 0)


def instantiate_template_files(d, agent_name):
    AOS_PATH = Path(__file__).parent
    for file_path in INIT_FILES:
        with open(AOS_PATH / file_path, "r") as fin:
            with open(d / file_path.name, "w") as fout:
                print(file_path)
                content = fin.read()
                now = datetime.now().strftime("%b %d, %Y %H:%M:%S")
                header = (
                    "# This file was auto-generated by `agentos init` "
                    f"on {now}."
                )
                fout.write(
                    content.format(
                        agent_name=agent_name,
                        conda_env=CONDA_ENV_FILE.name,
                        file_header=header,
                        abs_path=d.absolute(),
                        os_sep=os.sep,
                    )
                )


@agentos_cmd.command()
@click.argument("dir_names", nargs=-1, metavar="DIR_NAMES")
@click.option(
    "--agent-name",
    "-n",
    metavar="AGENT_NAME",
    default="BasicAgent",
    callback=validate_agent_name,
    help="This is used as the name of the MLflow Project and "
    "Conda env for all *Directory Agents* being created. "
    "AGENT_NAME may not contain ' ', ':', or '/'.",
)
@click.option(
    "--package-location",
    "-l",
    metavar="PACKAGE_LOCATION",
    type=click.Path(),
    default="./.acr",
    help="Path to AgentOS Component Registry installation directory",
)
def init(dir_names, agent_name, package_location):
    """Initialize current (or specified) directory as an AgentOS agent.

    \b
    Arguments:
        [OPTIONAL] DIR_NAMES zero or more space separated directories to
                             initialize. They will be created if they do
                             not exist.

    Creates an agent main.py file, a conda env, and an MLflow project file
    in all directories specified, or if none are specified, then create
    the files in current directory.
    """
    dirs = [Path(".")]
    if dir_names:
        dirs = [Path(d) for d in dir_names]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        curr_package_location = d / package_location
        create_agent_directory_structure(curr_package_location)
        create_core_data(curr_package_location)
        instantiate_template_files(d, agent_name)
        d = "current working directory" if d == Path(".") else d
        click.echo(
            f"Finished initializing AgentOS agent '{agent_name}' in {d}."
        )


def get_class_from_config(agent_dir_path, config):
    """Takes class_path of form "module.Class" and returns the class object."""
    sys.path.append(config["python_path"])
    file_path = agent_dir_path / Path(config["file_path"])
    spec = importlib.util.spec_from_file_location("TEMP_MODULE", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, config["class_name"])
    sys.path.pop()
    return cls


# TODO - uglily communicates to save_data() the dynamic data location
def decorate_save_data_fns(package_location):
    dl = get_data_location(package_location)
    agentos.__dict__["save_data"].__dict__["data_location"] = dl
    agentos.__dict__["restore_data"].__dict__["data_location"] = dl
    agentos.__dict__["save_tensorflow"].__dict__["data_location"] = dl
    agentos.__dict__["restore_tensorflow"].__dict__["data_location"] = dl


# TODO - V hacky!  is this the a reasonable way to go?
def restore_saved_data(package_location):
    decorate_save_data_fns(package_location)
    # TODO - REMOVE ME?
    # saved_data = {}
    # TODO - handle aliasing
    # files = data_location.glob("*")
    # for f in files:
    #    print(f"Restoring data at {f}")
    #    with open(f, "rb") as f_in:
    #        data = pickle.load(f_in)
    #    saved_data[f.name] = data
    # agentos.__dict__["saved_data"] = saved_data


def load_agent_from_path(agent_file, package_location):
    agent_path = Path(agent_file)
    agent_dir_path = agent_path.parent.absolute()
    config = configparser.ConfigParser()
    config.read(agent_path)

    restore_saved_data(package_location)

    agent_cls = get_class_from_config(agent_dir_path, config["Agent"])
    env_cls = get_class_from_config(agent_dir_path, config["Environment"])
    environment = env_cls(**config["Environment"])
    environment_spec = environment.get_spec()
    policy_cls = get_class_from_config(agent_dir_path, config["Policy"])
    policy = policy_cls(environment_spec=environment_spec, **config["Policy"])

    agent_kwargs = {
        "environment": environment,
        "policy": policy,
        **config["Agent"],
    }
    return agent_cls(**agent_kwargs)


def back_up_agent(agent, package_location):
    # TODO - dedup with restore_saved_data
    package_location = Path(package_location).absolute()
    data_location = get_data_location(package_location)
    backups_location = get_backups_location(package_location)
    shutil.copytree(data_location, backups_location / str(uuid.uuid4()))


def test_agent(test_episodes, agent_file, package_location, hz, max_iters):
    agent = load_agent_from_path(agent_file, package_location)
    print()
    print(
        f"Agent trained for {agent.get_step_count()} steps "
        f"({agent.get_episode_count()} episodes):"
    )
    _run(test_episodes, agent_file, package_location, hz, max_iters)
    print()


@agentos_cmd.command()
@click.option(
    "--iterations",
    "-i",
    type=click.INT,
    default=1,
    help="Number of episodes to run.",
)
@click.option(
    "--test-every",
    "-t",
    type=click.INT,
    default=0,
    help="Number of learning episodes between performance eval.",
)
@click.option(
    "--test-episodes",
    "-e",
    type=click.INT,
    default=0,
    help="Number of episodes to run for performance eval.",
)
@click.option(
    "--agent-file",
    "-f",
    type=click.Path(exists=True),
    default="./agent.ini",
    help="Path to agent definition file (agent.ini).",
)
@click.option(
    "--package-location",
    "-l",
    metavar="PACKAGE_LOCATION",
    type=click.Path(),
    default="./.acr",
    help="Path to AgentOS Component Registry installation directory",
)
@click.option(
    "--hz",
    "-h",
    metavar="HZ",
    default=None,
    type=int,
    help="Frequency to call agent.advance().",
)
@click.option(
    "--max-iterations",
    "-m",
    metavar="MAX_STEPS",
    type=int,
    default=None,
    help="Stop running agent after this many calls to advance().",
)
def learn(**kwargs):
    """Trains an agent by calling its learn() method in a loop."""
    iterations = kwargs["iterations"]
    test_every = kwargs["test_every"]
    test_episodes = kwargs["test_episodes"]
    agent_file = kwargs["agent_file"]
    package_location = kwargs["package_location"]
    hz = kwargs["hz"]
    max_iterations = kwargs["max_iterations"]
    agent = load_agent_from_path(agent_file, package_location)

    for i in range(iterations):
        if test_every and i % test_every == 0:
            back_up_agent(agent, package_location)
            test_agent(
                test_episodes, agent_file, package_location, hz, max_iterations
            )
        agent.learn()


@agentos_cmd.command()
@click.option(
    "--iterations",
    "-i",
    type=click.INT,
    default=1,
    help="Number of episodes to run.",
)
@click.option(
    "--agent-file",
    "-f",
    type=click.Path(exists=True),
    default="./agent.ini",
    help="Path to agent definition file (agent.ini).",
)
@click.option(
    "--package-location",
    "-l",
    metavar="PACKAGE_LOCATION",
    type=click.Path(),
    default="./.acr",
    help="Path to AgentOS Component Registry installation directory",
)
@click.option(
    "--hz",
    "-h",
    metavar="HZ",
    default=None,
    type=int,
    help="Frequency to call agent.advance().",
)
@click.option(
    "--max-iterations",
    "-m",
    metavar="MAX_STEPS",
    type=int,
    default=None,
    help="Stop running agent after this many calls to advance().",
)
def run(iterations, agent_file, package_location, hz, max_iterations):
    """Run an agent by calling advance() on it until it returns True"""
    _run(iterations, agent_file, package_location, hz, max_iterations)


def _run(iterations, agent_file, package_location, hz, max_iterations):
    all_steps = []
    for i in range(iterations):
        # TODO - A faster way to reset the agent isntead of reloading
        agent = load_agent_from_path(agent_file, package_location)
        steps = agentos.run_agent(agent, hz=hz, max_iters=max_iterations)
        all_steps.append(steps)

    if all_steps:
        mean = statistics.mean(all_steps)
        median = statistics.median(all_steps)
        print()
        print(f"Max steps over {iterations} trials: {max(all_steps)}")
        print(f"Mean steps over {iterations} trials: {mean}")
        print(f"Median steps over {iterations} trials: {median}")
        print(f"Min steps over {iterations} trials: {min(all_steps)}")


if __name__ == "__main__":
    agentos_cmd()
