import torch
import gym
from A2C import ActorCritic
import numpy as np
from matplotlib import pyplot as PLT
import multiprocessing as mp
import torch.optim
from torch.nn import functional as F
from Memory import Memory, Sample
import time
import os
from Map import Game, Scribe

import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(message)s')


def encode_state(state):
    map_size = 15
    quest_number = 5

    pos = state[0]
    cargo = state[1]
    gas = state[2]
    money = state[3]
    new_state = np.zeros(map_size * 2 + quest_number + 2, )
    new_state[pos[0]] = 1
    new_state[pos[1]] = 1
    for i in range(len(cargo)):
        new_state[map_size * 2 + i] = cargo[i]
    new_state[-2] = (gas / env.gas_max) ** 2
    new_state[-1] = np.clip(money / 500, 0, 1)

    return new_state


def init_memory(nr_of_runs):
    for i in range(nr_of_runs):
        run_episode(False)


def sliding_window(buffer, window_size=25):
    new_buffer = []
    for i in range(len(buffer) - window_size):
        new_buffer.append(np.sum(buffer[i:i + window_size]) / window_size)
    return new_buffer


def loss_fn(preds, r):
    return -1 * torch.sum(r * preds.cpu())


def plot_axis(line, data, ax):
    line.set_ydata(data)
    line.set_xdata(range(len(data)))
    ax.set_xlim([0, len(data)])
    min = np.min(data)
    max = np.max(data)
    ax.set_ylim([min - np.abs(min) * 0.1, max + np.abs(max) * 0.1])


def act(state):
    encoded_state = torch.from_numpy(state).float()
    # encoded_state = torch.from_numpy(encode_state(state)).float()

    encoded_state = encoded_state.to(GPU_DEVICE)
    policy = MasterBrain.actor_model(encoded_state)
    # try:
    # action = np.random.choice(np.array([i for i in range(action_count)]), p=policy.to(cpu).data.numpy())
    logits = policy.view(-1)
    action_dist = torch.distributions.Categorical(logits=logits)
    # rand = np.random.randint(0, 100)
    # # print(rand)
    # if rand < 1:
    #     print(action_dist.probs.view(-1).data)
    action = action_dist.sample().cpu().view(-1).numpy()[0]
    # except ValueError:
    #     print("policy:", policy)
    #     print("policy.data:", policy.to(cpu).data.numpy())
    return policy, action


def run_episode(should_replay=True):
    state = env.reset()
    values, probs, rewards = [], [], []
    total_rewards = 0
    steps = master_params["MAX_STEPS"]

    for i in range(master_params["MAX_STEPS"]):
        # env.render()
        policy, action = act(state)
        probs.append(policy[action])
        next_state, reward, is_done, into = env.step(action)
        # if is_done:
        #     reward = -10
        total_rewards = total_rewards + reward
        # memory.append(Sample(state, action, reward, policy.view(-1)[action].clone(), next_state, is_done))
        memory.append((state, action, reward, next_state, is_done))

        state = next_state
        # state = torch.from_numpy(encode_state(next_state)).float()
        if is_done:
            steps = i + 1
            break
    if should_replay:
        learn()

    # env.scribe.set_steps(steps)
    invalid.append(0)
    # invalid.append(env.scribe.percentage)
    total_steps.append(steps)
    reward_buffer.append(total_rewards / steps)
    return total_rewards


def learn():
    start = time.time()
    batch = memory.sample_batch(master_params["BATCH_SIZE"])
    rewards, states, advantages, critic_values, actor_probs = [], [], [], [], []
    for i, sample in enumerate(batch):
        state, action, reward, next_state, is_done = sample

        # state = encode_state(state)

        # tensor_state = torch.from_numpy(encode_state(state)).float()
        tensor_state = torch.from_numpy(state).float()
        tensor_state = tensor_state.to(GPU_DEVICE)


        probs = MasterBrain.actor_model(tensor_state).view(-1)[action]


        critic_value = MasterBrain.critic_model(tensor_state)
        # print(critic_value)
        if not is_done:
            future_state = MasterBrain.target_critic(torch.from_numpy(next_state).float().to(GPU_DEVICE)).cpu().data.numpy()[0]
            # future_state = MasterBrain.target_critic(torch.from_numpy(encode_state(next_state)).float().to(GPU_DEVICE))
            # print(future_state)
            # future_state = MasterBrain.target_critic(torch.from_numpy(encode_state(next_state)).float().to(GPU_DEVICE)).cpu().data.numpy()[0]
            reward = reward + future_state * master_params["discount"]

            # print(future_state)
        # print(critic_value)
        rewards.append(reward)
        states.append(state)
        critic_values.append(critic_value)
        actor_probs.append(probs)
        # print(critic_values[-1])


    actor_probs = torch.stack(actor_probs).flip(dims=(0, )).view(-1)

    critic_values = torch.stack(critic_values).flip(dims=(0, )).view(-1)

    # print(critic_values)

    # noinspection PyArgumentList
    rewards = torch.Tensor(rewards).flip(dims=(0, )).view(-1)

    # rewards = F.normalize(rewards, dim=0)

    # print("rewards", torch.mean(rewards))
    # print("critic_values", torch.mean(critic_values))

    advantages = rewards - critic_values.detach().cpu()
    actor_loss = loss_fn(actor_probs, advantages)
    # critic_loss = torch.abs(critic_values - rewards.to(GPU_DEVICE))
    # critic_loss = torch.abs(rewards.to(GPU_DEVICE) - critic_values)
    # critic_loss = torch.pow(critic_values, 2)
    # critic_loss = torch.pow(critic_values - rewards.to(GPU_DEVICE), 2)
    critic_loss = torch.pow(rewards.to(GPU_DEVICE) - critic_values, 2)
    # critic_loss = torch.pow(critic_values , 2)
    # critic_loss = torch.zeros(len(rewards), dtype=torch.double, device=GPU_DEVICE)
    # critic_loss = c_loss(critic_values - rewards.to(GPU_DEVICE), critic_loss)
    critic_loss = critic_loss.sum()



    MasterBrain.backpropagate_actor(actor_loss)
    MasterBrain.backpropagate_critic(critic_loss)


    losses.append(actor_loss.clone().detach())
    crit_losses.append(critic_loss.cpu().detach())
    TD_errors.append(torch.mean(advantages))





def save_to_file():
    torch.save(MasterBrain.actor_model.state_dict(), control_dict["ACTOR_PATH"])
    torch.save(MasterBrain.critic_model.state_dict(), control_dict["CRITIC_PATH"])
    logging.debug("Weights files updated")


def plot_and_save(filename, sliding_window_enhancer):
    avr_r = sliding_window(reward_buffer, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)
    avr_s = sliding_window(total_steps, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)
    avr_l = sliding_window(losses, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)
    avr_adv = sliding_window(TD_errors, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)
    avr_crit = sliding_window(crit_losses, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)
    avr_i = sliding_window(invalid, control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)



    plot_axis(line1, avr_r, ax1)
    plot_axis(line2, avr_s, ax2)
    plot_axis(line3, avr_l, ax3)
    plot_axis(line4, avr_adv, ax4)
    plot_axis(line5, avr_crit, ax5)
    plot_axis(line6, avr_i, ax6)

    PLT.savefig("Graphs\\" + filename)



if torch.cuda.is_available():
    GPU_DEVICE = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    logging.debug("Running on the GPU")
else:
    GPU_DEVICE = torch.device("cpu")
    logging.debug("Running on the CPU")


# env = Game()
# env = gym.make("MountainCar-v0")
env = gym.make("CartPole-v0")
# env = gym.make("Acrobot-v1")



losses = []
reward_buffer = []
total_steps = []
invalid = []
TD_errors = []
crit_losses = []


computing_time = []
backpropagation_time = []
learning_time = []

MODEL_PATH = os.path.abspath("Models")

c_loss = torch.nn.SmoothL1Loss()

cpu = torch.device("cpu")

memory = Memory(200000)

# brain = ActorCritic(4, 2)

# observation_count = env.observation_space
# observation_count = 37
# action_count = env.action_count

print(env.action_space)
print(env.observation_space)

observation_count = 4

action_count = 2

master_params = {
    'EPOCHS': 10000,
    'n_workers': 5,
    "actor_lr": 9e-5,
    "critic_lr": 9e-4,
    "BATCH_SIZE": 50,
    "discount": 0.99,
    "MAX_STEPS": 1500,
}


control_dict = {
    "RESET_TARGET_EVERY": 25,
    "SAVE_TO_PATH_EVERY": 50,
    "ACTOR_PATH": os.path.join(MODEL_PATH, "Actor_model.json"),
    "CRITIC_PATH": os.path.join(MODEL_PATH, "Critic_model.json"),
    "SLIDING_INIT_VALUE": 20
}


MasterBrain = ActorCritic(observation_count, action_count, GPU_DEVICE, actor_lr=master_params["actor_lr"],
                          critic_lr=master_params["critic_lr"])

init_memory(3)


fig = PLT.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

PLT.ion()


labels = ["Rewards", "Steps", "Actor loss", "Advantages", "Critic loss", "Invalid"]

averaged_rewards = sliding_window(reward_buffer)
averaged_steps = sliding_window(total_steps)
averaged_loss = sliding_window(losses)


line1, = ax1.plot(averaged_rewards, label=labels[0])
line2, = ax2.plot(averaged_steps, label=labels[1])
line3, = ax3.plot(averaged_loss, label=labels[2])
line4, = ax4.plot(TD_errors, label=labels[3])
line5, = ax5.plot(crit_losses, label=labels[4])
line6, = ax6.plot(invalid, label=labels[5])
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()
ax6.legend()


sliding_window_enhancer = 0

for i in range(1, master_params['EPOCHS']):
    reward = run_episode()
    torch.cuda.empty_cache()
    # if i % 100 == 0:
    logging.debug("".join(["episode ", str(i), " Reward: ", str(reward)]))
    # logging.debug(env.scribe)

    if i % control_dict["RESET_TARGET_EVERY"] == 0:
        MasterBrain.update_target()
    if i % control_dict["SAVE_TO_PATH_EVERY"] == 0:
        sliding_window_enhancer += 5
        save_to_file()
        plot_and_save("Graphs_iter_" + str(i) + ".png", sliding_window_enhancer)










# plot_axis(line1, total_rewards, ax1)
# plot_axis(line2, total_steps, ax2)
# plot_axis(line3, losses, ax3)

PLT.savefig("result.png")

