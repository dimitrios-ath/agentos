## Pain points 

* Developing AOS while developing components (e.g. componentizing r2d2 while
  AOS interface is unstable)
* I changed something, does my agent still work? (mini benchmark?)
* Developing multiple components simultaneously is a pain (working 5 different repos simultaneously--agentos, cartpole, r2d2_policy, r2d2_trainer, r2d2_dataset)

## Notes 7/30/21

* Trainer wants an env spec; pass during initialization
* Difficulty - sharing models between components (both policy and trainer want to ref the same network)
* Difficulty - trainer and dataset want to share derived parameters (sequence_length)
    * Soln: before_instantiation
    * Environment spec has to be in shared data (which is okay, because it can use shared data to initialize based on params) because dataset needs access
    * Environment.get_spec() is now a classmethod to support getting it before instantiation
    * Each component has global config and local config (?)
    * Another way - allow ordering of component initialization
    * Another another way - allow specification of requirements

* The top-level problem is components share dependencies (data, but also more complex structures).  How does AOS facillitate this sharing?
* Observation and transition adding is piped through the actor because we want the recurrent state



## TODO

// 1. See how hard it will be to commit to a custom spec
// 2. Maybe implement spec
// 3. Get agent working (maybe improve testing and training feedback)
4. Componentize R2D2

Killing and reloading the reverb server when we reload the agent...good or bad?
Reimplement Hz and max_iterations

* Logger object
* Unify run loops


### Possible Components

Trainer
?? Network
Dataset
Policy

Can we handle offline and distributed?


Tension between how expressivity/generality of agent.py versus how much is
baked into the AOS runtime as assumptions.  This choice affects how much code
one has to write to reuse agent. That is, agent.py can orchestrate a very
specific set of interactions with a specific set of components but that makes
it harder to swap components (or, at least, it makes the component APIs more
demanding).

## Notes

### 7/29/2021

branch: nj_componentize_r2d2

Had a meeting with Andy last night about AOS metrics and data management.
Fallout:

* Good progress on data management and metrics
* MLFlow can do all this (and more!) but it might not be perfectly adapted to
  RL
* Think hard before reimplementing much more MLFlow functionality
* BUT not urgent to switch the management/metrics backend to MLFlow


TODOs and TOTHINKS:

* Flesh out last user story a bit (about the component development workflow)
  (see next slide)
    * Some solutions might include programmatic agent instantiation for
      benchmarking
    * A lifecycle diagram might also help
* We might want an alternate demo of how to develop your own component
* Unify learning loops (e.g. run, learn, etc) so we can use a single
  metrics/debugging infrastructure (and record benchmark results during a run)
* // Wordsmithing output of run metrics
* What does MLFlow not address for RL?


### 7/24/2021:

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


### 7/22/2021

* Broke out save_data and save_tensorflow (and updated acme_r2d2)
* Added "core data" that tracks an agents training steps and episodes
* Now, we have flags that determine how often we backup and test and agent while training



## ETC

### Basic Data Saving Functionality to be ported MLFlow

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

### OLD TODOs

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



