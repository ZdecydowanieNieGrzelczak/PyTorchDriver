import sys

import torch
import gym
from A2C import ActorCritic, StateOracle
import numpy as np
from matplotlib import pyplot as PLT
import multiprocessing as mp
import torch.optim
from torch.nn import functional as F
from Memory import Memory, Sample
import os
from ConvEnv import ConvEnv, Scribe
import json
import pickle
from torch.nn.utils.rnn import pad_sequence


import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(message)s')

def create_config_dict():
    config = {}
    learning = {}
    environment = {}
    learning["discount"] = master_params["discount"]
    learning["batch size"] = master_params["BATCH_SIZE"]
    learning["actor learning rate"] = master_params["actor_lr"]
    learning["critic learning rate"] = master_params["critic_lr"]
    learning["actor shapes"] = MasterBrain.actor_shapes[0]
    learning["critic shapes"] = MasterBrain.critic_shapes[0]
    learning["memory"] = {"size": memory.size, "prioritized": memory.is_prioritized}
    learning["is_cargo_in"] = env.observation_space != (env.width * env.height + 2)

    environment["stations"] = {"nr": env.station_nr, "uniform": env.uniform_gas_stations}
    environment["quest"] = {"nr": env.station_nr, "prepaid": env.prepaid, "reward per step": env.reward_per_step}
    environment["step penalty"] = -env.gas_price
    environment["death penalty"] = -env.death_reward
    environment["gas"] = {"start": env.start_gas, "end": env.gas_max}
    environment["size"] = {"width": env.width, "height": env.height}
    environment["codes"] = {"player": env.player_code, "station": env.station_code, "quest": env.quest_code, "reward": env.reward_code}
    environment["normalized rewards"] = env.normalize_reward
    if env.normalize_reward:
        environment["reward normalizer"] = env.reward_normalizer


    config["learning"] = learning
    config["environment"] = environment

    with open('config.json', 'w') as fp:
        json.dump(config, fp)


def to_oracle_input(tensor_state, action):
    actions = np.zeros(shape=action_count)
    actions[action] = 1
    tensor_actions = torch.from_numpy(actions).float().to(GPU_DEVICE)
    return torch.cat((tensor_state, tensor_actions), dim=0)

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
    state = env.reset()
    i = 0
    while i < nr_of_runs:
        action = env.sample_move()
        next_state, reward, is_done = env.step(action)
        if reward != 0:
            priority_memory.append((state, action, 0.25, reward, next_state, is_done))
        else:
            memory.append((state, action, 0.25, reward, next_state, is_done))
        state = next_state
        if is_done:
            i += 1
            state = env.reset()

def sliding_window(buffer, window_size=25):
    new_buffer = []
    for i in range(len(buffer) - window_size):
        new_buffer.append(np.sum(buffer[i:i + window_size]) / window_size)
    return new_buffer


def loss_fn(preds, r):
    # return torch.sum(r * preds.cpu())
    return r * preds.cpu()


def plot_axis(line, data, ax):
    line.set_ydata(data)
    line.set_xdata(range(len(data)))
    ax.set_xlim([0, len(data)])
    min = np.min(data)
    max = np.max(data)
    ax.set_ylim([min - np.abs(min) * 0.1, max + np.abs(max) * 0.1])


def act(state):
    if exploration_dict["FORCED_EXPLORATION"]:
        if exploration_dict["currently_exploring"]:
            if np.random.random() <= current_epsilon:
                return env.sample_move(), 0.25

    # return env.sample_move(), 0.25

    # print(state)
    # print(np.shape(state))

    encoded_state = torch.from_numpy(state).float()
    # encoded_state = torch.from_numpy(encode_state(state)).float()

    encoded_state = encoded_state.to(GPU_DEVICE)
    policy = MasterBrain.actor_model(encoded_state)
    logits = policy.view(-1)
    action_dist = torch.distributions.Categorical(logits=logits)

    # print(action_dist.probs.data)

    action = action_dist.sample().cpu().view(-1).numpy()[0]

    prob = action_dist.probs[action]

    return action, prob.cpu().data.numpy()


def run_episode():
    state = env.reset()
    total_rewards = 0
    steps = master_params["MAX_STEPS"]
    probs = []

    for i in range(master_params["MAX_STEPS"]):
        # env.render()
        action, prob = act(state)
        next_state, reward, is_done = env.step(action)
        probs.append(prob)

        total_rewards = total_rewards + reward
        if reward != 0:
            priority_memory.append((state, action, prob, reward, next_state, is_done))
        else:
            memory.append((state, action, prob, reward, next_state, is_done))

        state = next_state
        if is_done:
            steps = i + 1
            break

    learn()

    env.scribe.set_steps(steps)
    # invalid.append(0)
    invalid.append(env.scribe.percentage)
    total_steps.append(steps)
    reward_buffer.append(total_rewards)
    TD_errors.append(np.mean(probs))
    # reward_buffer.append(total_rewards / steps)

    return total_rewards


def learn():

    batch = memory.sample_batch(master_params["BATCH_SIZE"])
    ISWeights = [1 for i in range(master_params["BATCH_SIZE"] + master_params["PRIORITY_BATCH_SIZE"])]

    priority_batch = priority_memory.sample_batch(master_params["PRIORITY_BATCH_SIZE"])

    batch = batch + priority_batch

    np.random.shuffle(batch)


    rewards, advantages, critic_values, actor_probs = [], [], [], []
    true_states, oracle_predicts = [], []
    ratio, entropy_list = [], []
    critic_targetes = []
    total_crit_loss, total_actor_loss = 0, 0
    for i, sample in enumerate(batch):
        # print(sample)
        state, action, old_prob, reward, next_state, is_done = sample
        ISWeight = ISWeights[i]

        # tensor_state = torch.from_numpy(encode_state(state)).float()
        tensor_state = torch.from_numpy(state).float()
        tensor_state = tensor_state.to(GPU_DEVICE)

        tensor_next_state = torch.from_numpy(next_state).float().to(GPU_DEVICE)

        # probs = MasterBrain.actor_model(tensor_state).view(-1)[action]
        policy = MasterBrain.actor_model(tensor_state)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        entropy = action_dist.entropy().mean().cpu()

        probs = action_dist.probs[action]

        # if PPO_dict["IMPLEMENT_PPO"]:
        #     # print("probs: ", probs.data, " old probs: ", old_prob)
        #     ratio.append((probs.cpu() - old_prob).exp())

        critic_value = MasterBrain.critic_model(tensor_state)
        if not is_done:
            future_state = MasterBrain.critic_model(tensor_next_state).cpu().data.numpy()[0]
            target_state = MasterBrain.target_critic(tensor_next_state).detach()
            # future_state = MasterBrain.target_critic(torch.from_numpy(encode_state(next_state)).float().to(GPU_DEVICE))
            target_state = reward * target_state * master_params["discount"]
            reward = reward + future_state * master_params["discount"]
        else:
            target_state = torch.Tensor([reward]).to(GPU_DEVICE)


        if curiosity_dict["IS_CURIOUS"]:
            true_states.append(tensor_next_state)
            # if should_oracle:
            predict = stateOracle(to_oracle_input(tensor_state, action))
            oracle_predicts.append(predict)
            # else:
            #     oracle_predicts.append(tensor_next_state)

        if PPO_dict["IMPLEMENT_PPO"]:

            if curiosity_dict["IS_CURIOUS"]:
                if should_oracle:
                    oracle_losses = o_loss(predict, tensor_next_state) * ISWeight
                    reward = reward + oracle_losses.cpu().detach() / curiosity_dict["oracle_decrease"]
                    advantages.append(oracle_losses.cpu().data.numpy() / curiosity_dict["oracle_decrease"])
                    if oracle_losses >= curiosity_dict["minimal_loss"]:
                        oracle_optim.zero_grad()
                        oracle_losses.backward(retain_graph=True)
                        oracle_optim.step()
                else:
                    advantages.append(0)

            advantage = reward - critic_value.cpu()



            ratio = (probs.cpu().detach() - old_prob).exp()
            alpha = PPO_dict["PPO_RATIO"]
            loss1 = ratio * advantage
            loss2 = torch.clamp(ratio, 1 - alpha, 1 + alpha) * advantage
            actor_loss = torch.min(loss1, loss2) * ISWeight # + entropy * master_params["ENTROPY_LR"]
            critic_huber_loss = c_loss(critic_value, target_state.detach()) * ISWeight

            MasterBrain.backpropagate_actor(actor_loss)
            MasterBrain.backpropagate_critic(critic_huber_loss)

            total_crit_loss += critic_huber_loss.cpu().data.numpy()
            total_actor_loss += actor_loss.cpu().data.numpy()


        if not PPO_dict["IMPLEMENT_PPO"]:
            rewards.append(reward)
            critic_values.append(critic_value)
            actor_probs.append(probs)
            critic_targetes.append(target_state)
            entropy_list.append(entropy)
        else:
            losses.append(total_actor_loss)
            crit_losses.append(total_crit_loss)


    if not PPO_dict["IMPLEMENT_PPO"]:

        actor_probs = torch.stack(actor_probs).flip(dims=(0, )).view(-1)

        ISWeights = torch.Tensor(ISWeights).flip(dims=(0, )).view(-1)
        critic_values = torch.stack(critic_values).flip(dims=(0, )).view(-1)
        critic_targetes = torch.stack(critic_targetes).flip(dims=(0, )).view(-1)

        # noinspection PyArgumentList
        rewards = torch.Tensor(rewards).flip(dims=(0, )).view(-1)



        if curiosity_dict["IS_CURIOUS"]:
            oracle_predicts = pad_sequence(oracle_predicts)
            true_states = pad_sequence(true_states)
            oracle_losses = o_loss(oracle_predicts, true_states)
            oracle_losses = oracle_losses.sum(dim=0) / curiosity_dict["oracle_decrease"] * ISWeights
            # print(oracle_losses.mean())
            # print("Average oracle bonus = ", torch.mean(oracle_losses).data.cpu())
            TD_errors.append(torch.mean(oracle_losses).data.cpu())
            rewards = rewards + oracle_losses.cpu().detach()
            oracle_loss = torch.sum(oracle_losses)
            oracle_optim.zero_grad()
            oracle_loss.backward(retain_graph=True)
            oracle_optim.step()


        advantages = rewards - critic_values.detach().cpu()

        # if PPO_dict["IMPLEMENT_PPO"]:
        #     ratio = torch.stack(ratio)
        #     entropy_list = torch.stack(entropy_list)
        #
        #     print("ratio min: ", ratio.min())
        #     print("ratio max: ", ratio.max())
        #
        #     alpha = PPO_dict["PPO_RATIO"]
        #     loss1 = ratio * advantages
        #     loss2 = torch.clamp(ratio, 1 - alpha, 1 + alpha) * advantages
        #     actor_loss = torch.min(loss1, loss2) # + entropy_list * master_params["ENTROPY_LR"]
        #     actor_loss = actor_loss.sum()
        total_actor_loss = loss_fn(actor_probs, advantages)

        if memory.is_prioritized:
            memory.update_nodes(np.abs(total_actor_loss.cpu().detach().numpy()))

        actor_loss = -1 * torch.sum(total_actor_loss * ISWeights)


        # critic_loss = torch.pow(critic_targetes.detach() - critic_values, 2) # TAK WIKIPEDIA NAWET MOWI


        critic_huber_loss = c_loss(critic_values, critic_targetes.detach()) * ISWeights.to(GPU_DEVICE)


        # critic_loss = torch.abs(critic_targetes.detach() - critic_values) # TAK WIKIPEDIA NAWET MOWI


        # critic_loss = F.normalize(critic_loss, dim=0)


        # critic_loss = critic_loss.sum()
        critic_loss = critic_huber_loss.sum()

        MasterBrain.backpropagate_actor(actor_loss)
        if torch.abs(critic_loss) > master_params["min_loss"]:
            MasterBrain.backpropagate_critic(critic_loss)


        losses.append(actor_loss.cpu().detach())
        crit_losses.append(critic_loss.cpu().detach())
        if not curiosity_dict["IS_CURIOUS"]:
            pass
            # TD_errors.append(torch.mean(actor_probs.cpu().data))
            # TD_errors.append(torch.mean(advantages))





def save_to_file():
    torch.save(MasterBrain.actor_model.state_dict(), control_dict["ACTOR_PATH"])
    torch.save(MasterBrain.critic_model.state_dict(), control_dict["CRITIC_PATH"])
    logging.debug("Weights files updated")


def plot_and_save(filename, sliding_window_enhancer):
    global last_index

    last_index -= (control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer)

    averaged_rewards.extend(sliding_window(reward_buffer[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))
    averaged_steps.extend(sliding_window(total_steps[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))
    averaged_loss.extend(sliding_window(losses[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))
    averaged_advantaged.extend(sliding_window(TD_errors[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))
    averaged_crit_losses.extend(sliding_window(crit_losses[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))
    averaged_invalid.extend(sliding_window(invalid[last_index:], control_dict["SLIDING_INIT_VALUE"] + sliding_window_enhancer))


    last_index = len(reward_buffer)

    plot_axis(line1, averaged_rewards, ax1)
    plot_axis(line2, averaged_steps, ax2)
    plot_axis(line3, averaged_loss, ax3)
    plot_axis(line4, averaged_advantaged, ax4)
    plot_axis(line5, averaged_crit_losses, ax5)
    plot_axis(line6, averaged_invalid, ax6)

    Graphs_path  = os.path.abspath("Graphs")

    PLT.savefig(filename)

    print("Graphs saved")


def save_buffers_binary():

    Graphs_path  = os.path.abspath("Graphs")

    with open('Rewards.p', 'wb') as fp:
        pickle.dump(reward_buffer, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Actor loss.p', 'wb') as fp:
        pickle.dump(losses, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Steps.p', 'wb') as fp:
        pickle.dump(total_steps, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Advantages.p', 'wb') as fp:
        pickle.dump(TD_errors, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Critic losses.p', 'wb') as fp:
        pickle.dump(crit_losses, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open('Invalid.p', 'wb') as fp:
        pickle.dump(invalid, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print("Buffers dumped")



def plot_trend(data, window_size=50):
    trend_analysis = PLT.figure(2)
    head = data[:window_size]
    tail = data[-window_size:]

    start_value = np.mean(np.asarray(head))
    end_value = np.mean(np.asarray(tail))

    step = (end_value - start_value) / (len(data) - window_size)

    trend_data = [start_value + i * step for i in range(window_size // 2, len(data) - (window_size // 2))]

    slided = sliding_window(data, window_size)

    x_data = [i for i in range(window_size // 2, len(data) - (window_size // 2))]
    x_full = [i for i in range(len(data))]


    PLT.plot(x_data, trend_data, label="trend line")
    PLT.plot(x_data, slided, label="averaged values")
    # PLT.plot(x_full, data, label="raw data")

    PLT.legend()

    PLT.savefig("trend visualisation.png")



if torch.cuda.is_available():
    GPU_DEVICE = torch.device("cuda:0")
    logging.debug("Running on the GPU")
else:
    GPU_DEVICE = torch.device("cpu")
    logging.debug("Running on the CPU")

if __name__ == '__main__':
    # env = Game()
    # env = gym.make("MountainCar-v0")
    # env = gym.make("CartPole-v0")
    # env = gym.make("Acrobot-v1")

    memory_dict = {
        "size": 524288,
        "current_beta": 0.2,
        "max_beta": 0.8,
        "starting_beta": 0.2,
        "memory_init_runs": 5
    }


    env = ConvEnv(quest_nr=4, station_nr=6, width=10, height=10, uniform_gas_stations=True, normalize_rewards=True)


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

    c_loss = torch.nn.SmoothL1Loss(reduction='none')
    o_loss = torch.nn.SmoothL1Loss(reduction="none")

    cpu = torch.device("cpu")

    memory = Memory(memory_dict["size"])
    priority_memory = Memory(100000)
    #
    # memory = PriorMemory(memory_dict["size"], is_multiprocessing=True)


    # brain = ActorCritic(4, 2)

    observation_count = env.observation_space
    # observation_count = 37
    action_count = env.action_count


    # observation_count = env.observation_space
    # action_count = env.action_count

    # observation_count = 4
    # action_count = 2


    master_params = {
        'EPOCHS': 40010,
        'n_workers': 5,
        "actor_lr": 1e-4,
        "critic_lr": 5e-4,
        "BATCH_SIZE": 256,
        "PRIORITY_BATCH_SIZE": 32,
        "discount": 0.97,
        "MAX_STEPS": 2500,
        "ENTROPY_LR": 1e-4,
        "min_loss": 0.00001
    }


    control_dict = {
        "RESET_TARGET_EVERY": 300,
        "SAVE_TO_PATH_EVERY": 10000,
        "ACTOR_PATH": os.path.join(MODEL_PATH, "Actor_model.json"),
        "CRITIC_PATH": os.path.join(MODEL_PATH, "Critic_model.json"),
        "SLIDING_INIT_VALUE": 5000,
        "CONTINUE_LEARNING": True,
        "SLIDING_INCREMENTAL_VALUE": 0,
        "PLOT_TREND_EVERY": 2000,
        "TEST_VALUES": False,
        "SAVE_BUFFERS_EVERY": 10000,
        "RANDOM_EPISODES_EVERY": 5,
        "NUMBER_IF_RANDOM_EPISODES": 2
    }

    exploration_dict = {
        "FORCED_EXPLORATION": False,
        "EXPLORATION_TIME": 500,
        "EXPLOITATION_TME": 500,
        "STARTING_EPSILON": 0.9,
        "ENDING EPSILONE": 0.0001,
        "currently_exploring": False,
    }


    curiosity_dict = {
        "IS_CURIOUS": False,
        "oracle_lr": 9e-4,
        "minimal_loss": 1e-5,
        "oracle_decrease": 128
    }

    PPO_dict = {
        "IMPLEMENT_PPO": True,
        "PPO_RATIO": 0.2
    }




    current_epsilon = exploration_dict["STARTING_EPSILON"]

    MasterBrain = ActorCritic(observation_count, action_count, GPU_DEVICE, actor_lr=master_params["actor_lr"],
                              critic_lr=master_params["critic_lr"])


    stateOracle = StateOracle(observation_count, action_count).to(GPU_DEVICE)


    oracle_optim = torch.optim.Adam(params=stateOracle.parameters(), lr=curiosity_dict["oracle_lr"])


    if control_dict["CONTINUE_LEARNING"]:
        MasterBrain.actor_model.load_state_dict(torch.load(control_dict["ACTOR_PATH"], map_location=torch.device('cpu')))
        MasterBrain.actor_model.eval()
        MasterBrain.critic_model.load_state_dict(torch.load(control_dict["CRITIC_PATH"], map_location=torch.device('cpu')))
        MasterBrain.critic_model.eval()
        MasterBrain.update_target()
        logging.debug("LOADED VALUES")


    init_memory(memory_dict["memory_init_runs"])


    fig = PLT.figure(1)



    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    PLT.ion()


    labels = ["Rewards", "Steps", "Actor loss", "Advantages", "Critic loss", "Invalid"]

    averaged_rewards = []
    averaged_steps = []
    averaged_loss = []
    averaged_advantaged = []
    averaged_crit_losses = []
    averaged_invalid = []

    last_index = (control_dict["SLIDING_INIT_VALUE"] + control_dict["SLIDING_INCREMENTAL_VALUE"])

    line1, = ax1.plot(reward_buffer, label=labels[0])
    line2, = ax2.plot(total_steps, label=labels[1])
    line3, = ax3.plot(losses, label=labels[2])
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


    if control_dict["TEST_VALUES"]:
        MasterBrain.critic_model.load_state_dict(torch.load(control_dict["CRITIC_PATH"]))
        MasterBrain.critic_model.eval()
        logging.debug("LOADED CRITIC VALUES")
        state = env.reset()
        # state[-2] = 0.000000001
        # state[-1] = 1
        values = np.zeros(shape=(env.width, env.height))
        for i in range(env.width):
            for j in range(env.height):
                previous = state[j + i * env.height]
                state[j + i * env.height] = env.player_code / 255
                # print(state[:-2].reshape(env.width, env.height))
                # print("")
                # print(j + i * env.height)
                logs = MasterBrain.actor_model(torch.from_numpy(state).float().to(GPU_DEVICE)).cpu().data
                action_dist = torch.distributions.Categorical(logits=logs)

                values[j][i] = np.argmax(action_dist.probs.data.numpy())

                # values[j][i] = MasterBrain.critic_model(torch.from_numpy(state).float().to(GPU_DEVICE)).cpu().data.numpy()
                state[j + i * env.height] = previous
        print(state[:-2].reshape(env.width, env.height))
        print(values)
        sys.exit()


    create_config_dict()
    save_buffers_binary()

    for i in range(1, master_params['EPOCHS']):
        reward = run_episode()
        torch.cuda.empty_cache()
        logging.debug("".join(["episode ", str(i), " Reward: ", str(reward)]))

        memory_dict["current_beta"] += (memory_dict["max_beta"] - memory_dict["starting_beta"]) / master_params['EPOCHS']

        if exploration_dict["FORCED_EXPLORATION"]:
            if exploration_dict["currently_exploring"]:
                logging.debug("Exploring with probability: " + str(current_epsilon))
        logging.debug(env.scribe)

        if i % control_dict["RESET_TARGET_EVERY"] == 0:
            MasterBrain.update_target()
        if i % control_dict["SAVE_TO_PATH_EVERY"] == 0:
            sliding_window_enhancer += control_dict["SLIDING_INCREMENTAL_VALUE"]
            save_to_file()
            plot_and_save("Graphs_" + str(i) + ".png", sliding_window_enhancer)
        if exploration_dict["FORCED_EXPLORATION"]:
            current_epsilon = (exploration_dict["STARTING_EPSILON"] - exploration_dict["ENDING EPSILONE"]) / \
                              exploration_dict["EXPLORATION_TIME"] * \
                (exploration_dict["EXPLORATION_TIME"] - (i % exploration_dict["EXPLOITATION_TME"]))

            if i % exploration_dict["EXPLOITATION_TME"] == 0:
                exploration_dict["currently_exploring"] = not exploration_dict["currently_exploring"]
        if i % control_dict["SAVE_BUFFERS_EVERY"] == 0:
            save_buffers_binary()
        if i % control_dict["RANDOM_EPISODES_EVERY"] == 0:
            init_memory(control_dict["NUMBER_IF_RANDOM_EPISODES"])

    PLT.savefig("result.png")
    plot_trend(reward_buffer)
    save_buffers_binary()
