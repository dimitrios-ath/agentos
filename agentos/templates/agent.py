{file_header}
import agentos


# A basic agent.
class {agent_name}(agentos.Agent):
    DEFAULT_ENTRY_POINT = "evaluate"

    def learn(self, num_episodes=10, test_every=True, test_num_episodes=5):
            """Set up some default params for this entry point."""
            super().learn(num_episodes, test_every, test_num_episodes)
