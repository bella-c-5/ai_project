# models resume improvement as an MDP
# resume section = state
# actions = add, improve, or skip

# contains a list of ordered states: objective + skills + projects + achievements
# evaluates what action likely improves the resume, returns the reward for that action, transitions to the next state

class ResumeMDP:
    def __init__(self):
        self.states = ["objective", "skills", "projects", "achievements"]
        self.actions = ["add", "improve", "skip"]
        self.rewards = {
            "add": 10,
            "improve": 5,
            "skip": 0
        }

    def reward(self, action):
        return self.rewards[action]

    def transition(self, state, action):
        idx = self.states.index(state)
        next_idx = min(idx + 1, len(self.states) - 1)
        return self.states[next_idx]