// 1. See how hard it will be to commit to a custom spec
// 2. Maybe implement spec
// 3. Get agent working (maybe improve testing and training feedback)
4. Componentize R2D2




Trainer
Network
Dataset
Policy


Offline
Distributed


Tension between how expressivity/generality of agent.py (coordinating the
interactions) and how much code one has to write to reuse agent.

ie agent.py can orchestrate a very specific set of interactions with a specific
set of components but that makes it harder to swap components

Notes 7/24/2021:

agentos default agent works with acme_r2d2

```
agentos init .
agentos install acme_r2d2
agentos run
agentos learn
```

TODO:
* Current difficulty: why does r2d2 insist on float input?
* **Often you get unlucky with initialization and run/learn on corridor does
  not terminate**
* Port others in registry
* What is this reverb checkpoint restore and is it interfering poorly with agentos

* **Componentize**


Design:
* Just included dm_env in agentos core, bad idea?


Notes 7/22/2021:

* Broke out save_data and save_tensorflow (and updated acme_r2d2)
* Added "core data" that tracks an agents training steps and episodes
* Now, we have flags that determine how often we backup and test and agent while training


## Basic Data Saving Functionality to be ported MLFlow

https://github.com/agentos-project/agentos/compare/master...nickjalbert:nj_componentize_r2d2?expand=1

* "Core data": core metrics important to an RL run (steps/episodes/reward in
  training, steps/reward in testing)
    * Probably should be integrated into the agent/AOS runtime by default

* Way to save/load/restore-if-exists core data
* Way to save/load/restore-if-exists auxillary data
* Way to save/load/restore-if-exists models (tensorflow or sonnet)
* Some standardized way to push debugging info out of AOS runtime/core
* Way to backup and rollback an agent
    * Incrementally through a long training process

* Some way to assess agent performance at different versions through time

make all envs work with r2d2

## General TODOs

Registry url: https://raw.githubusercontent.com/nickjalbert/agentos/nj_registry/registry.yaml

- An RL glossary/wiki - good non-technical intros to RL and common algos

- Ashvin's idea @learnable decorator for functions

- Environment helper functions:
    - valid_actions
    - current_obs

- Kwargs from agent.ini

- Policy action_space and observation_space

- A lot of frameworks are trying to be everything to everyone
    - What if we just made a REALLY strong framework to enable a specific
      technique



