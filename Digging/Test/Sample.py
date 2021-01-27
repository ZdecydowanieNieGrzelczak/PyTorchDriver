class Sample:
    def __init__(self, action, action_probs, reward, episode_dict):
        self.actions = action
        self.action_probs = action_probs
        self.reward = reward
        self.episode_dict = episode_dict

    def __cmp__(self, other):
        if self.reward < other:
            return False
        return True