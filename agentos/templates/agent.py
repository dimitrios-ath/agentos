{file_header}
import agentos


# A basic agent.
class {agent_name}(agentos.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs = self.environment.reset()

    def learn(self):
        print("{agent_name} is calling self.policy.improve()")
        step_count = self.get_step_count()
        episode_count = self.get_episode_count()
        action = None
        self.obs = self.environment.reset()
        reward = None
        done = False
        info = {{}}
        self.policy.observe(action, self.obs, reward, done, info)
        while not done:
            action = self.policy.decide(
                self.obs, self.environment.valid_actions
            )
            self.obs, reward, done, info = self.environment.step(action)
            self.policy.observe(action, self.obs, reward, done, info)
            step_count += 1
        self.policy.improve()
        episode_count += 1
        self.save_step_count(step_count)
        self.save_episode_count(episode_count)

    def advance(self):
        print("{agent_name} is taking an action")
        next_action = self.policy.decide(
            self.obs, self.environment.valid_actions
        )
        self.obs, reward, done, info = self.environment.step(next_action)
        print("\tObservation: " + str(self.obs))
        print("\tReward:      " + str(reward))
        print("\tDone:        " + str(done))
        print("\tInfo:        " + str(info))
        print()
        return done

    def get_step_count(self):
        return agentos.restore_data('step_count')
        
    def get_episode_count(self):
        return agentos.restore_data('episode_count')

    def save_step_count(self, step_count):
        return agentos.save_data('step_count', step_count)
        
    def save_episode_count(self, episode_count):
        return agentos.save_data('episode_count', episode_count)


