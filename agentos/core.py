"""Core AgentOS APIs."""
import time
from threading import Thread
from collections import namedtuple
import pickle


class MemberInitializer:
    """Takes all constructor kwargs and sets them as class members.

    For example, if MyClass is a MemberInitializer:

    a = MyClass(foo='bar')
    assert a.foo == 'bar'
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Agent(MemberInitializer):
    """An Agent observes and takes actions in its environment till done.

    An agent holds an environment ``self.env``, which it can use
    to observe and act by calling ``self.env.step()`` that takes
    an observation and returns an action.

    Like a human, an Agent lives in a stream of time. To to bring an
    Agent to life (i.e. have it exist and take actions in its
    environment), simply call agent.advance() repeatedly until
    that function returns True (which means the agent is done).

    The agent can maintain any sort of state (e.g., a policy for
    deciding its next action), but any use or updates of state must
    happen from within the agent's advance() function (which itself can
    be arbitrarily complex, call other functions, etc.).

    Often, an agent's advance function has 3 phases:

        1) pre-action
        2) take action and save observation
        3) post-action

    ...with phases 1 and 3 often including internal decision making,
    learning, use of models, state updates, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.curr_obs = None
        self._should_reset = True

    def step(self):
        """Takes one action within the environment"""
        if self._should_reset:
            self.curr_obs = self.environment.reset()
            self._should_reset = False
            self.dataset.add(None, None, self.curr_obs, None, None, {})
        action = self.policy.decide(
            self.curr_obs, self.environment.valid_actions
        )
        prev_obs = self.curr_obs
        self.curr_obs, reward, done, info = self.environment.step(action)
        self.dataset.add(prev_obs, action, self.curr_obs, reward, done, info)
        if done:
            self._should_reset = True
        return prev_obs, action, self.curr_obs, reward, done, info

    def rollout(self, should_learn):
        """Does training on one rollout worth of transitions"""
        done = False
        step_count = 0
        while not done:
            _, _, _, _, done, _ = self.step()
            step_count += 1
        if should_learn:
            self.trainer.improve(self.dataset, self.policy)
            prev_step_count = self.get_step_count()
            prev_episode_count = self.get_episode_count()
            self.save_step_count(prev_step_count + step_count)
            self.save_episode_count(prev_episode_count + 1)
        return step_count

    def advance(self):
        """Returns True when agent is done; False or None otherwise."""
        raise NotImplementedError

    def get_step_count(self):
        return restore_data("step_count")

    def get_episode_count(self):
        return restore_data("episode_count")

    def save_step_count(self, step_count):
        return save_data("step_count", step_count)

    def save_episode_count(self, episode_count):
        return save_data("episode_count", episode_count)


class Policy(MemberInitializer):
    """Pick next action based on last observation from environment.

    Policies are used by agents to encapsulate any state or logic necessary
    to decide on a next action given the last observation from an env.
    """

    def __init__(self, **kwargs):
        pass

    def decide(self, observation, actions, should_learn=False):
        """Takes an observation and returns next action to take.

        :param observation: should be in the `observation_space` of the
            environments that this policy is compatible with.
        :param should_learn: should the agent learn from the transition?
        :returns: action to take, should be in `action_space` of the
            environments that this policy is compatible with.
        """
        raise NotImplementedError


class Trainer(MemberInitializer):
    def improve(self, dataset, policy):
        pass


class Dataset(MemberInitializer):
    def add(self, prev_obs, action, curr_obs, reward, done, info):
        pass


# Inspired by OpenAI's gym.Env
# https://github.com/openai/gym/blob/master/gym/core.py
class Environment(MemberInitializer):
    """Minimalist port of OpenAI's gym.Env."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = None
        self.observation_space = None
        self.reward_range = None

    def step(self, action):
        """Perform the action in the environment."""
        raise NotImplementedError

    def reset(self):
        """Resets the environment to an initial state."""
        raise NotImplementedError

    def render(self, mode):
        raise NotImplementedError

    def close(self, mode):
        pass

    def seed(self, seed):
        raise NotImplementedError


# https://github.com/deepmind/sonnet#tensorflow-checkpointing
# TODO - custom saver/restorer functions
# TODO - aliasing
# TODO - V hacky way to pass in the global data location; we decorate
#        this function with the location in restore_saved_data in cli.py
# TODO - ONLY works for the demo (Acme on TF) because the dynamic module
#        loading in ACR core breaks pickle. Need to figure out a more general
#        way to handle this
def save_tensorflow(name, network):
    print("Saving module")
    import tensorflow as tf

    checkpoint = tf.train.Checkpoint(module=network)
    checkpoint.save(save_tensorflow.data_location / name)


# https://github.com/deepmind/sonnet#tensorflow-checkpointing
# Same caveats as save_data above
def restore_tensorflow(name, network):
    import tensorflow as tf

    checkpoint = tf.train.Checkpoint(module=network)
    latest = tf.train.latest_checkpoint(restore_tensorflow.data_location)
    if latest is not None:
        print("AOS: Restoring policy network from checkpoint")
        checkpoint.restore(latest)
    else:
        print("AOS: No checkpoint found for policy network")


def save_data(name, data):
    with open(save_data.data_location / name, "wb") as f:
        pickle.dump(data, f)


def restore_data(name):
    with open(restore_data.data_location / name, "rb") as f:
        return pickle.load(f)


def run_agent(agent, hz=40, max_iters=None, as_thread=False):
    """Run an agent, optionally in a new thread.

    If as_thread is True, agent is run in a thread, and the
    thread object is returned to the caller. The caller may
    need to call join on that that thread depending on their
    use case for this agent_run.

    :param agent: The agent object you want to run
    :param hz: Rate at which to call agent's `advance` function. If None,
        call `advance` repeatedly in a tight loop (i.e., as fast as possible).
    :param max_iters: Maximum times to call agent's `advance` function,
        defaults to None.
    :param as_thread: Set to True to run this agent in a new thread, defaults
        to False.
    :returns: Either a running thread (if as_thread=True) or None.
    """

    def runner():
        done = False
        iter_count = 0
        while not done:
            if max_iters and iter_count >= max_iters:
                break
            done = agent.advance()
            if hz:
                time.sleep(1 / hz)
            iter_count += 1
        return iter_count

    if as_thread:
        t = Thread(target=runner)
        t.start()
        return t
    else:
        return runner()


def default_rollout_step(policy, obs, step_num):
    """
    The default rollout step function is the policy's decide function.

    A rollout step function allows a developer to specify the behavior
    that will occur at every step of the rollout--given a policy
    and the last observation from the env--to decide
    what action to take next. This usually involves the rollout's
    policy and may perform learning. It also, may involve using, updating,
    or saving learning related state including hyper-parameters
    such as epsilon in epsilon greedy.

    You can provide your own function with the same signature as this default
    if you want to have a more complex behavior at each step of the rollout.
    """
    return policy.decide(obs)


def rollout(policy, env_class, step_fn=default_rollout_step, max_steps=None):
    """Perform rollout using provided policy and env.

    :param policy: policy to use when simulating these episodes.
    :param env_class: class to instantiate an env object from.
    :param step_fn: a function to be called at each step of rollout.
        The function can have 2 or 3 parameters, and must return an action:

        * 2 parameter definition: policy, observation.
        * 3 parameter definition: policy, observation, step_num.

        Default value is ``agentos.core.default_rollout_step``.

    :param max_steps: cap on number of steps per episode.
    :return: the trajectory that was followed during this rollout.
        A trajectory is a named tuple that contains the initial observation (a
        scalar) as well as the following arrays: actions, observations,
        rewards, dones, contexts. The ith entry of each array corresponds to
        the action taken at the ith step of the rollout, and the respective
        results returned by the environment after taking that action. To learn
        more about the semantics of these, see the documentation and code of
        gym.Env.
    """
    actions = []
    observations = []
    rewards = []
    dones = []
    contexts = []

    env = env_class()
    obs = env.reset()
    init_obs = obs
    done = False
    step_num = 0
    while True:
        if done or (max_steps and step_num >= max_steps):
            break
        if step_fn.__code__.co_argcount == 2:
            action = step_fn(policy, obs)
        elif step_fn.__code__.co_argcount == 3:
            action = step_fn(policy, obs, step_num)
        else:
            raise TypeError("step_fn must accept 2 or 3 parameters.")
        obs, reward, done, ctx = env.step(action)
        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        contexts.append(ctx)
        step_num += 1
    Trajectory = namedtuple(
        "Trajectory",
        [
            "init_obs",
            "actions",
            "observations",
            "rewards",
            "dones",
            "contexts",
        ],
    )
    return Trajectory(
        init_obs, actions, observations, rewards, dones, contexts
    )


def rollouts(
    policy,
    env_class,
    num_rollouts,
    step_fn=default_rollout_step,
    max_steps=None,
):
    """
    :param policy: policy to use when simulating these episodes.
    :param env_class: class to instatiate an env object from.
    :param num_rollouts: how many rollouts (i.e., episodes) to perform
    :param step_fn: a function to be called at each step of each rollout.
                    The function can have 2 or 3 parameters.
                    2 parameter definition: policy, observation.
                    3 parameter definition: policy, observation, step_num.
                    The function must return an action.
    :param max_steps: cap on number of steps per episode.
    :return: array with one namedtuple per rollout, each tuple containing
             the following arrays: observations, rewards, dones, ctxs
    """
    return [
        rollout(policy, env_class, step_fn, max_steps)
        for _ in range(num_rollouts)
    ]


# https://github.com/deepmind/acme/blob/master/acme/specs.py
EnvironmentSpec = namedtuple(
    "EnvironmentSpec", ["observations", "actions", "rewards", "discounts"]
)
