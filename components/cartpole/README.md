# CartPole
A thin wrapper around the CartPole implementation in OpenAI's Gym toolkit

## Installation

Requires Python 3.6 to 3.8.

* Create a virtualenv, e.g. `virtualenv -p /usr/local/opt/python\@3.8/bin/python3.8 env`
* Activate your virtualenv: `source env/bin/activate`
* Clone the latest [AgentOS](https://github.com/agentos-project/agentos) master
* `pip install -e [path/to/agentos/clone/]`
* Run the test: `python cartpole.py`
* Format code: `python scripts/format_code.py`
* Lint code: `python scripts/lint_code.py`
