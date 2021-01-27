import os

from A2C import ActorCritic
from ConvEnv import *
import numpy as np
import torch
import pickle
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(message)s')

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


def act(state):
    encoded_state = torch.from_numpy(state).float()

    encoded_state = encoded_state.to(GPU_DEVICE)
    policy = brain.actor_model(encoded_state)
    logits = policy.view(-1)
    action_dist = torch.distributions.Categorical(logits=logits)


    action = action_dist.sample().cpu().view(-1).numpy()[0]

    probs = action_dist.probs.cpu().data.numpy()

    return action, probs

def run_episode():
    episode_dict = {}
    state, episode_dict["setup"] = env.reset()
    episode_dict["Changes"] = {}
    total_rewards = 0
    actions = []
    action_probs = []
    is_done = False
    steps = 0
    while not is_done:
        action, probs = act(state)
        next_state, reward, is_done, change_dict = env.step(action)
        if change_dict is not None:
            episode_dict["Changes"][steps] = change_dict
        actions.append(action)
        probs_string = []
        for prob in probs:
            probs_string.append('{0:.16f}'.format(prob))
        action_probs.append(probs_string)
        total_rewards = total_rewards + reward
        state = next_state
        steps += 1
    return total_rewards, actions, action_probs, steps, episode_dict

GPU_DEVICE = torch.device("cuda:0")

env = ConvEnv(width=10, height=10, station_nr=6, quest_nr=4, uniform_gas_stations=True, normalize_rewards=True)

brain = ActorCritic(env.observation_space, env.action_count, GPU_DEVICE)

actor_path = os.path.abspath("Digging") + "\\Actor_model11.json"

brain.actor_model.load_state_dict(torch.load(actor_path))
brain.actor_model.eval()

NUMBER_OF_TRIES = 50000
curently_best = -10
currently_best_sample = None

minimum = 10

best_samples = []

for i in range(NUMBER_OF_TRIES):
    rewards, actions, action_probs, steps, episode_dict = run_episode()
    if rewards > curently_best:
        curently_best = rewards
        currently_best_sample = Sample(actions, action_probs, rewards, episode_dict)
    if rewards > minimum:
        best_samples.append(Sample(actions, action_probs, rewards, episode_dict))
    if i % 50 == 0:
        logging.debug(str(i) + " out of " + str(NUMBER_OF_TRIES))
        logging.debug("currently best: " + str(round(curently_best, 1)))


with open('Digging\\Samples.p', 'wb') as fp:
    pickle.dump(best_samples, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('Digging\\BestSample.p', 'wb') as fp:
    pickle.dump(currently_best_sample, fp, protocol=pickle.HIGHEST_PROTOCOL)