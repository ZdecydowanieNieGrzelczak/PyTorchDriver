import torch
import math
import numpy as np
from torch import nn
from torch import optim

import torch.multiprocessing as mp






class ActorCritic(nn.Module):
    def __init__(self, observation_count, action_count, device, actor_shapes=(1200, 600), critic_shapes=(1000, 500), actor_lr=0.0009,
                 critic_lr=1e-4):
        super(ActorCritic, self).__init__()
        self.observation_count = observation_count
        self.action_count = action_count
        self.actor_shapes = actor_shapes
        self.critic_shapes = critic_shapes
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device

        self.actor_model, self.actor_optimizer = self.create_actor_model(actor_shapes)
        self.critic_model, self.critic_optimizer = self.create_critic_model(critic_shapes)

        self.target_critic, self.target_critic_optim = self.create_critic_model(critic_shapes)

    def create_actor_model(self, shapes):

        model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_count, shapes[0]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(shapes[0], shapes[1]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(shapes[1], self.action_count),
            torch.nn.LogSoftmax()
        )

        model = model.to(self.device)

        adam = torch.optim.Adam(model.parameters(), lr=self.actor_lr)
        sdg = torch.optim.SGD(model.parameters(), lr=self.actor_lr)

        return model, adam

    def create_critic_model(self, shapes):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.observation_count, shapes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(shapes[0], shapes[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(shapes[1], 1),
            # torch.nn.Tanh()
        )

        model = model.to(self.device)
        # model = model.cuda()

        adam = optim.Adam(model.parameters(), lr=self.critic_lr)

        return model, adam

    def backpropagate_actor(self, actor_loss):

        before = []
        after = []

        for params in self.actor_model.parameters():
            before.append(torch.sum(params))
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        for params in self.actor_model.parameters():
            after.append(torch.sum(params))


    def backpropagate_critic(self, critic_loss):
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()



    def update_target(self):
        self.target_critic.load_state_dict(self.critic_model.state_dict())
        sums = []
        for param in self.actor_model.parameters():
            sums.append(torch.sum(param))
