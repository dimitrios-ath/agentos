"""
AgentOS AgentRunManager provides an API that agents can use to log and retrieve
agent runs and run-related data/stats/tags/etc. The two primary types of
runs used by agents are Learning and Evaluation runs.

The agent run manager also contains the logic for tracking the lineage of
learning runs so that a model's training history is captured, and publishable.
"""
from pathlib import Path
import statistics
from typing import Optional
from collections import namedtuple
from agentos.component import Component
from agentos.run import Run
from agentos.registry import Registry


_EPISODE_KEY = "episode_count"
_STEP_KEY = "step_count"

_RUN_STATS_MEMBERS = [
    _EPISODE_KEY,
    _STEP_KEY,
    "max_reward",
    "median_reward",
    "mean_reward",
    "min_reward",
    "training_episode_count",
    "training_step_count",
]

RunStats = namedtuple("RunStats", _RUN_STATS_MEMBERS)


class AgentRunManager:
    """
    A Component used to manage Agent training and evaluation runs.

    Provides RunManagers, which are context managers that can be used
    by agents when executing episodes as part of evaluation or learning
    with something like:

         with self.run_manager.evaluation_run as run_manager:
              # run an episode
              run_manager.log_episode(
                    # episode_data
                    ...
              )
    """

    LEARN_KEY = "learn"
    RESET_KEY = "reset"
    RESTORE_KEY = "restore"
    EVALUATE_KEY = "evaluate"
    RUN_TYPE_TAG = "run_type"
    AGENT_NAME_KEY = "agent_name"
    ENV_NAME_KEY = "environment_name"

    def __init__(self, *args, **kwargs):
        self.episode_data = []
        self._active_run = Run.active_run(self)

        def evaluate_run_manager(
            agent_name: str = None, environment_name: str = None
        ) -> RunContextManager:
            return RunContextManager(
                run_manager=self,
                run_type=self.EVALUATE_KEY,
                agent_name=agent_name,
                environment_name=environment_name,
            )

        def learn_run_manager(
            agent_name: str = None, environment_name: str = None
        ) -> RunContextManager:
            return RunContextManager(
                run_manager=self,
                run_type=self.LEARN_KEY,
                agent_name=agent_name,
                environment_name=environment_name,
            )

        self.evaluate_run = evaluate_run_manager
        self.learn_run = learn_run_manager
        self.run_type = None

    def log_run_type(self, run_type: str) -> None:
        self.run_type = run_type
        self._active_run.set_tag(self.RUN_TYPE_TAG, self.run_type)

    def log_agent_name(self, agent_name: str) -> None:
        self._active_run.log_param(self.AGENT_NAME_KEY, agent_name)

    def log_environment_name(self, environment_name: str) -> None:
        self._active_run.log_param(self.ENV_NAME_KEY, environment_name)

    def _log_run_type(self) -> None:

    def log_run_metrics(self):
        assert self.episode_data, "No episode data!"
        run_stats = self._get_run_stats()
        for key in _RUN_STATS_MEMBERS:
            val = getattr(run_stats, key)
            self._active_run.log_metric(key, val)

    def get_training_info(self) -> (int, int):
        print("IN TRAINING INFO")
        runs = self._active_run.get_all_runs()
        print(type(runs))
        total_episodes = 0
        total_steps = 0
        for run in runs:
            if run.data.tags.get(self.RUN_TYPE_TAG) == self.LEARN_KEY:
                total_episodes += int(run.data.metrics.get(_EPISODE_KEY, 0))
                total_steps += int(run.data.metrics.get(_STEP_KEY, 0))
        return total_episodes, total_steps

    def print_results(self):
        if not self.episode_data:
            return
        run_stats = self._get_run_stats()
        if self.run_type == self.LEARN_KEY:
            print(
                "\nTraining results over "
                f"{run_stats.episode_count} episodes:"
            )
            print(
                "\tOverall agent was trained on "
                f"{run_stats.training_step_count} transitions over "
                f"{run_stats.training_episode_count} episodes"
            )
        else:
            print(
                "\nBenchmark results over "
                f"{run_stats.episode_count} episodes:"
            )
            print(
                "\tBenchmarked agent was trained on "
                f"{run_stats.training_step_count} transitions over "
                f"{run_stats.training_episode_count} episodes"
            )
        print(
            f"\tMax reward over {run_stats.episode_count} episodes: "
            f"{run_stats.max_reward}"
        )
        print(
            f"\tMean reward over {run_stats.episode_count} episodes: "
            f"{run_stats.mean_reward}"
        )
        print(
            f"\tMedian reward over {run_stats.episode_count} episodes: "
            f"{run_stats.median_reward}"
        )
        print(
            f"\tMin reward over {run_stats.episode_count} episodes: "
            f"{run_stats.min_reward}"
        )
        print()

    def _get_run_stats(self):
        run_id = Run.active_run(self, fail_if_no_active_run=True).identifier
        run_data = [d for d in self.episode_data if d["active_run"] == run_id]
        episode_lengths = [d["steps"] for d in run_data]
        episode_returns = [d["reward"] for d in run_data]
        training_episodes, training_steps = self.get_training_info()
        return RunStats(
            episode_count=len(run_data),
            step_count=sum(episode_lengths),
            max_reward=max(episode_returns),
            mean_reward=statistics.mean(episode_returns),
            median_reward=statistics.median(episode_returns),
            min_reward=min(episode_returns),
            training_episode_count=training_episodes,
            training_step_count=training_steps,
        )

    def add_episode_data(self, steps: int, reward: float):
        self.episode_data.append(
            {
                "steps": steps,
                "reward": reward,
                "active_run": Run.active_run(self).identifier,
            }
        )

    def reset(self) -> None:
        self.log_run_type(self.RESET_KEY)


class RunContextManager:
    def __init__(
        self,
        run_manager: AgentRunManager,
        run_type: str,
        agent_name: Optional[str],
        environment_name: Optional[str],
    ):
        self.run_manager = run_manager
        self.run_type = run_type
        self.agent_name = agent_name or "agent"
        self.environment_name = environment_name or "environment"
        self._check_component_exists_in_run(self.agent_name)
        self._check_component_exists_in_run(self.environment_name)

    def _check_component_exists_in_run(self, role_type: str) -> None:
        run = Run.active_run(self.run_manager)
        artifacts_dir = run.download_artifacts('.')
        spec_path = Path(artifacts_dir) / Run.RUN_COMMAND_REGISTRY_FILENAME
        names = [
            Component.Identifier.from_str(c_id).name
            for c_id in Registry.from_yaml(spec_path)
            .get_component_specs()
            .keys()
        ]
        expected_name = getattr(self, f"{role_type}_name")
        if expected_name not in names:
            print(
                f"Warning: unknown {role_type.capitalize()} component: "
                f"{expected_name}.  Run will not be publishable."
            )
            self.components_exist = False
            Run.run_output.log_param(f"{role_type}_exists", False)
        else:
            Run.run_output.log_param(f"{role_type}_exists", True)

    def __enter__(self) -> None:
        self.run_manager.log_run_type(self.run_type)
        self.run_manager.log_agent_name(self.agent_name)
        self.run_manager.log_environment_name(self.environment_name)

    def __exit__(self, type, value, traceback) -> None:
        self.run_manager.log_run_metrics()
        self.run_manager.print_results()
