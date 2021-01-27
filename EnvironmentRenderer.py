import time

import pygame

import ConvEnv


class Tile:

    def __init__(self, x, y, code):
        self.x = x
        self.y = y
        self.code = code

class ScreenRenderer:

    color_dict = {
        "playground": (230, 227, 220),
        "stats": (92, 99, 115),
        "player": (0, 0, 0),
        2137: (13, 13, 13)
    }

    player_pos = [0, 0]
    running = True
    SCREEN_SIZE = (830, 620)
    previous = None

    def __init__(self, environment):
        self.env = environment
        self.fill_color_dict()
        self.quests_codes = [self.env.quest_code + i for i in range(self.env.quest_nr)]
        self.rewards = [self.env.reward_code + i for i in range(self.env.quest_nr)]

        pygame.init()
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        self.create_background()
        # self.display()
        pygame.display.flip()

    def fill_color_dict(self):
        for i in range(self.env.quest_nr):
            self.color_dict[i + self.env.quest_code] = (0, 150 + i * 5, 0)
            self.color_dict[i + self.env.reward] = (0, 0, 150 + i * 5)
        self.color_dict[self.env.station_code] = (140, 0, 0)
        self.color_dict[0] = (169, 217, 194)

    def create_background(self):
        pygame.draw.rect(self.screen, self.color_dict["playground"], pygame.Rect(10, 10, 600, 600))
        pygame.draw.rect(self.screen, self.color_dict["stats"], pygame.Rect(620, 10, 200, 600))
        self.create_map()

    def create_map(self):
        map = self.env.map

        self.player_pos = self.env.player_pos
        x, y = self.player_pos
        self.previous = Tile(x, y, map[x][y])
        for i, row in enumerate(map):
            print(row)
            for j, tile in enumerate(row):
                color = self.color_dict[tile]
                self.draw_rect(j, i, color)
        # self.draw_player(y, x)

    def draw_rect(self, x, y, color):
        pygame.draw.rect(self.screen, color, pygame.Rect(15 + x * 40, 15 + y * 40, 30, 30))

    def draw_player(self, x, y):
        self.previous = Tile(x, y, self.env.map[x][y])
        pygame.draw.rect(self.screen, self.color_dict["player"], pygame.Rect(15 + x * 40, 20 + y * 40, 30, 15))
        pygame.draw.circle(self.screen, self.color_dict["player"], (22 + x * 40, 35 + y * 40), 6)
        pygame.draw.circle(self.screen, self.color_dict["player"], (38 + x * 40, 35 + y * 40), 6)

    def move_player(self, dir_vector, current_gas, current_money, action_probs, action, new_tile):
        self.draw_rect()

    def display(self):
        while self.running:
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    self.running = False



env = ConvEnv.ConvEnv(quest_nr=0, station_nr=6, width=10, height=10, uniform_gas_stations=True)

ScreenRenderer(env)


while True:
    time.sleep(1)
