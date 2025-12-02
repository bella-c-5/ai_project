# Models resume improvement as a simple MDP:
# - States = major resume sections
# - Actions = add/improve/skip
# - Rewards = importance of each action
# App.py will use this to generate actionable suggestions

class ResumeMDP:
    def __init__(self):
        self.states = ["objective", "skills", "projects", "achievements"]
        self.actions = ["add", "improve", "skip"]
        self.rewards = {
            "add": 10,
            "improve": 5,
            "skip": 0
        }

    # Reward for chosen action
    def reward(self, action):
        return self.rewards[action]

     # Simple next-state transition - move through ordered sections
    def transition(self, state, action):
        idx = self.states.index(state)
        next_idx = min(idx + 1, len(self.states) - 1)
        return self.states[next_idx]