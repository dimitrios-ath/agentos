import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.absolute()
EXAMPLE_AGENT_PATH = REPO_ROOT / "example_agents"
DEV_REQS_PATH = REPO_ROOT / "dev-requirements.txt"
RLLIB_REQS_PATH = EXAMPLE_AGENT_PATH / "rllib_agent" / "requirements.txt"
SB3_REQS_PATH = EXAMPLE_AGENT_PATH / "sb3_agent" / "requirements.txt"
ACME_DQN_REQS_PATH = EXAMPLE_AGENT_PATH / "acme_dqn" / "requirements.txt"
ACME_R2D2_REQS_PATH = EXAMPLE_AGENT_PATH / "acme_r2d2" / "requirements.txt"
WEB_REQS_PATH = REPO_ROOT / "web" / "requirements.txt"


def install_requirements():
    answer = "n"
    if len(sys.argv) > 1 and sys.argv[1] == "-y":
        print("-y passed; will not confirm installation")
        answer = "y"
    else:
        msg = (
            "This will install all dev requirements and example agent "
            "Python requirements into the currently active virtualenv.  "
            "Continue [y/n]? "
        )
        answer = input(msg)

    if answer.lower() not in ["y", "yes"]:
        print("Aborting...")
        sys.exit(0)

    subprocess.run(["pip", "install", "-r", DEV_REQS_PATH])
    if sys.platform == "linux":
        # Get CPU-only version of torch in case CUDA is not proper configured
        subprocess.run(
            [
                "pip",
                "install",
                "-r",
                RLLIB_REQS_PATH,
                "-f",
                "https://download.pytorch.org/whl/torch_stable.html",
            ]
        )
        subprocess.run(["pip", "install", "-r", ACME_DQN_REQS_PATH])
        subprocess.run(["pip", "install", "-r", ACME_R2D2_REQS_PATH])
    else:
        subprocess.run(["pip", "install", "-r", RLLIB_REQS_PATH])
    subprocess.run(["pip", "install", "-r", SB3_REQS_PATH])
    subprocess.run(["pip", "install", "-r", WEB_REQS_PATH])
    subprocess.run(["pip", "install", "-e", REPO_ROOT])


if __name__ == "__main__":
    install_requirements()
