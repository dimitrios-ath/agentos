"""
Run flake8 linter on AgentOS Python files

To use::

  $ python scripts/lint_code.py
"""

import os
import sys
from pathlib import Path
from subprocess import run
from subprocess import PIPE

from shared import root_dir
from shared import traverse_tracked_files

returncode = 0

IGNORED_FILES = [
    "agentos/templates/agent.py",
    "agentos/templates/policy.py",
    "agentos/templates/environment.py",
    "agentos/templates/dataset.py",
    "agentos/templates/trainer.py",
]


def flake_file(path):
    global returncode
    extension = os.path.splitext(path)[1]
    if extension != ".py":
        return
    cmd = ["flake8", "--max-line-length", "79", path]
    result = run(cmd, stdout=PIPE)
    returncode = returncode | result.returncode
    out = result.stdout.decode("utf-8")
    if out:
        print(path)
        print(out)
        print()


if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        path = Path(arg).absolute()
        flake_file(path)
else:
    traverse_tracked_files(root_dir, flake_file, IGNORED_FILES)
sys.exit(returncode)
