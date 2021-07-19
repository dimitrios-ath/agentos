{file_header}
import agentos


# A basic agent.
class {agent_name}(agentos.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs = self.environment.reset()

    def learn(self):
        print("{agent_name} is calling self.policy.improve()")
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
        self.policy.improve()

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
