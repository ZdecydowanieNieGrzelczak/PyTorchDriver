import random, time
import numpy as np
from math import trunc


class Scribe:

    def __init__(self, nr_of_quests, nr_of_stations, nr_of_locations, is_total=False):
        self.picked = np.zeros(shape=nr_of_quests)
        self.ended = np.zeros(shape=nr_of_quests)
        self.tanked = np.zeros(shape=nr_of_stations)
        self.mobility = np.zeros(shape=nr_of_locations)
        self.died = np.zeros(shape=nr_of_locations)
        self.is_total = is_total
        self.invalid_action = 0
        self.steps = 0
        self.percentage = 0

    def __str__(self):
        if self.is_total:
            representation = "During whole game:\n"
        else:
            representation = "During run:\n"
        picked = "Picked: {0:.0f} Q1: {p[0]:.0f} Q2: {p[1]:.0f} Q3: {p[2]:.0f} Q4: {p[3]:.0f} Q5: {p[4]:.0f}\n".format(np.sum(self.picked), p=self.picked)
        ended = "Ended:  {0:.0f} Q1: {p[0]:.0f} Q2: {p[1]:.0f} Q3: {p[2]:.0f} Q4: {p[3]:.0f} Q5: {p[4]:.0f}\n".format(np.sum(self.ended), p=self.ended)
        tanked = "Tanked: {0:.0f} LP1: {p[0]:.0f} LP2: {p[1]:.0f} LP3: {p[2]:.0f}\n".format(np.sum(self.tanked), p=self.tanked)
        if self.steps == 0:
            invalid = "Invalid actions taken: {0}\n".format(self.invalid_action)
        else:
            invalid = "Invalid actions taken: {0} | {1}\nPercentage:{2:.2f}%".format(self.invalid_action, self.steps,
                                                                                     self.percentage)

        return ''.join([representation, picked, ended, tanked, invalid])

    def __add__(self, other):
        self.picked += other.picked
        self.ended += other.ended
        self.tanked += other.tanked
        self.died += other.died
        self.invalid_action += other.invalid_action

    def set_steps(self, steps):
        self.steps = steps
        self.percentage = self.invalid_action / self.steps * 100




class Game:

    wait_discount = 1
    gas_max = 300
    map_size = 15
    prepaid = 0.25
    gas_price = 1
    reward_per_step = 8
    large_quest_bonus = 1.03
    random_deviation = 0.2
    quest_multiplier = 3

    def __init__(self, reward_normalizer=600, random_map=False, quest_nr=5, station_nr=3):
        self.quest_nr = quest_nr
        self.station_nr = station_nr
        self.reward_normalizer = reward_normalizer
        random.seed(time.time())
        self.gas = 150
        self.money = 400
        self.map = []
        self.player_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        self.is_done = False
        self.reward = 0
        if not random_map:
            self.quests = [[12, 6], [0, 2], [11, 11], [5, 4], [4, 13]]
            self.destinations = [[6, 7], [2, 5], [13, 2], [5, 10], [2, 2]]
            self.rewards = [300, 200, 400, 250, 500]
            self.gas_stations = [[7, 7], [0, 5], [10, 6]]
            self.cargo= [0, 0, 0, 0]
        else:
            self.set_random_points()
        self.has_tanked = False
        self.started = False
        self.ended = False
        self.action_space = (0, 1, 2, 3)
        self.state_count = self.map_size * 2 + 1 + len(self.quests) + 1
        self.observation_space = 37
        # self.actions = [self.action_up, self.action_down, self.action_left, self.action_right]
        self.scribe = Scribe(self.quest_nr, self.station_nr, self.map_size ** 2)

        self.actions = [self.action_up, self.action_down, self.action_left, self.action_right, self.action_special]
        self.action_count = len(self.action_space)
        # self.actions = [self.action_up, self.action_down, self.action_left, self.action_right, self.action_wait, self.action_special]

        for i in range(self.map_size):
            temp = []
            for y in range(self.map_size):
                temp.append("XX")
            self.map.append(temp)

        self.map[0][0] = "ST"

        for i in range(len(self.quests)):
            quest = self.quests[i]
            self.map[quest[0]][quest[1]] = "Q" + str(i + 1)

        for i in range(len(self.destinations)):
            destination = self.destinations[i]
            self.map[destination[0]][destination[1]] = "R" + str(i + 1)

        for gas_station in self.gas_stations:
            self.map[gas_station[0]][gas_station[1]] = "LP"

    def reset(self):
        self.player_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        self.gas = 150
        self.money = 400
        self.cargo = [0, 0, 0, 0, 0]
        self.is_done = False
        self.reward = 0
        self.has_tanked = False
        self.started = False
        self.ended = False
        self.scribe = Scribe(self.quest_nr, self.station_nr, self.map_size ** 2)
        return self.get_state_object()

    def get_state_object(self):
        state = (self.player_pos, self.cargo, self.gas, self.money)
        # print("State ", state)
        return state

    def step(self, action):
        if action > len(self.action_space) - 1:
            raise Exception('InvalidAction', action)
        else:
            reward = self.actions[action]()
            # print("player pos: ", self.player_pos)
            if self.gas <= 0:
                reward -= self.reward_normalizer * 0.5
                self.is_done = True
                self.scribe.died[self.player_pos[0] + self.player_pos[1] * 15] += 1

        reward += self.action_special()
        # reward /= self.reward_normalizer
        state = (self.get_state_object(), reward, self.is_done, [])
        return state

    def action_up(self):
        if self.player_pos[0] == 0:
            return self.action_wait()
        else:
            self.player_pos[0] -= 1

            self.gas -= 1
            return - 1 * self.gas_price

    def action_down(self):
        if self.player_pos[0] == 14:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[0] += 1
            return - 1 * self.gas_price

    def action_right(self):
        if self.player_pos[1] == 14:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] += 1
            return - 1 * self.gas_price

    def action_left(self):
        if self.player_pos[1] == 0:
            return self.action_wait()
        else:
            self.gas -= 1
            self.player_pos[1] -= 1
            return - 1 * self.gas_price

    def action_wait(self):
        self.scribe.invalid_action += 1
        self.gas -= 1 * self.wait_discount
        return -1 * self.wait_discount * self.gas_price

    def action_special(self):
        if self.player_pos in self.quests:
            quest = self.quests.index(self.player_pos)
            if self.cargo[quest] == 0:
                self.scribe.picked[quest] += 1
                self.started = True
                self.cargo[quest] = 1
                self.money += self.prepaid * self.rewards[quest]
                return self.prepaid * self.rewards[quest] * self.quest_multiplier

        elif self.player_pos in self.destinations:
            quest = self.destinations.index(self.player_pos)
            if self.cargo[quest] == 1:
                self.scribe.ended[quest] += 1
                self.ended = True
                self.cargo[quest] = 0
                self.money += (1 - self.prepaid) * self.rewards[quest]
                return (1 - self.prepaid) * self.rewards[quest] * self.quest_multiplier

        if self.player_pos in self.gas_stations:
            cost = self.gas - self.gas_max
            self.has_tanked = True
            self.scribe.tanked[self.gas_stations.index(self.player_pos)] += 1
            if abs(cost) > self.money:
                cost = self.money * -1
                self.money = 0
                self.gas -= cost
            else:
                self.money += cost
                self.gas = self.gas_max
            # print("Tanking. Cost: ", cost)
            return 0
        # self.scribe.invalid_action += 1
        # return 0
        # return self.action_wait()
        return 0

    def sample_move(self):
        return random.randint(0, len(self.action_space) - 1)

    def print_map(self, map=None):
        print("")
        print("Actions: 0: up, 1: down, 2: left, 3: right")
        print("")


        for i in range(self.map_size):
            line = "     "
            for j in range(self.map_size):
                if map is None:
                    line += self.map[i][j] + " "
                else:
                    line += "X" + str(trunc(map[i][j])) + " "
            print(line)

    def set_random_points(self):
        points = []
        quest = []
        destinations = []
        gas_stations = []
        rewards = []
        for i in range(self.quest_nr * 2):
            x = random.randint(self.map_size)
            y = random.randint(self.map_size)
            while [x, y] in points:
                x = random.randint(self.map_size)
                y = random.randint(self.map_size)
            points.append([x, y])
            if i < self.quest_nr:
                quest.append([x, y])
            else:
                destinations.append([x, y])

        for i in range(self.quest_nr):
            manhattan_distance = np.sum(np.abs([quest[i][0] - destinations[i][0], quest[i][1] - destinations[i][1]]))
            reward = self.reward_per_step * (self.large_quest_bonus ** manhattan_distance)
            reward *= random.random(self.random_deviation) - random.random(self.random_deviation * 2)
            rewards.append(reward)

        for i in range(self.station_nr):
            x = random.randint(self.map_size)
            y = random.randint(self.map_size)
            while [x, y] in points:
                x = random.randint(self.map_size)
                y = random.randint(self.map_size)
            points.append([x, y])
            gas_stations.append([x, y])

        self.quests = quest
        self.destinations = destinations
        self.rewards = rewards
        self.gas_stations = gas_stations
        self.cargo = [0 for i in range(self.quest_nr)]




