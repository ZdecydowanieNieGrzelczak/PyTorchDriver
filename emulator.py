import sys

from ConvEnv import ConvEnv
from A2C import ActorCritic
import os
import numpy as np
import torch

MODEL_PATH = os.path.abspath("Models")

control_dict = {
    "RESET_TARGET_EVERY": 75,
    "SAVE_TO_PATH_EVERY": 150,
    "ACTOR_PATH": os.path.join(MODEL_PATH, "Actor_model.json"),
    "CRITIC_PATH": os.path.join(MODEL_PATH, "Critic_model.json"),
    "SLIDING_INIT_VALUE": 120,
    "CONTINUE_LEARNING": False,
    "SLIDING_INCREMENTAL_VALUE": 0,
    "PLOT_TREND_EVERY": 150,
    "TEST_VALUES": False
}

game = ConvEnv(width=5, height=5, quest_nr=1, station_nr=1)

device = torch.device("cpu")

# brain = ActorCritic(game.observation_space, game.action_count, device)
#
#
# brain.actor_model.load_state_dict(torch.load(control_dict["ACTOR_PATH"]))
# brain.actor_model.eval()
# brain.critic_model.load_state_dict(torch.load(control_dict["CRITIC_PATH"]))
# brain.critic_model.eval()


# exploration_dict = {
#     "FORCED_EXPLORATION": True,
#     "EXPLORATION_TIME": 500,
#     "EXPLOITATION_TME": 500,
#     "STARTING_EPSILON": 0.9,
#     "ENDING EPSILONE": 0.0001,
#     "currently_exploring": False,
# }
#
#
# for i in range(20000):
#     current_epsilon = (exploration_dict["STARTING_EPSILON"] - exploration_dict["ENDING EPSILONE"]) / \
#                               exploration_dict["EXPLORATION_TIME"] * \
#                 (exploration_dict["EXPLORATION_TIME"] - (i % exploration_dict["EXPLOITATION_TME"]))
#     print(current_epsilon)




while True:
    state = game.reset()
    total_rewards = 0
    is_done = False
    while not is_done:
        print(state[:-2].reshape(5, 5))
        print("money: ", state[-1])
        print("gas: ", state[-2])
        print("rewards: ", total_rewards)
        action = input("Enter action 1: up, 2: down 3: left 4: right: ")
        action = int(action) - 1
        if action == 5:
            sys.exit()
        if action == 4:
            break
        next_state, reward, is_done, into = game.step(action)
        total_rewards = total_rewards + reward

        state = next_state
        if is_done:
            break


