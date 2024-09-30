import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time

# Hiperparámetros
GAMMA = 0.99               # Factor de descuento
LR = 0.001                 # Tasa de aprendizaje
EPSILON_START = 1.0        # Valor inicial de epsilon
EPSILON_MIN = 0.01         # Valor mínimo de epsilon
EPSILON_DECAY = 0.995      # Factor de decaimiento de epsilon
BATCH_SIZE = 64            # Tamaño de lote
MEMORY_SIZE = 10000        # Tamaño del buffer de experiencia
TARGET_UPDATE = 10         # Frecuencia de actualización de la red target

# Definición de la red neuronal
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Clase para manejar el algoritmo DQN
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        # Red principal y red target
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

    # Política epsilon-greedy
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return q_values.argmax().item()

    # Almacenar en el buffer de replay
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Entrenar el modelo con Experience Replay
    def replay_experience(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Muestrear un batch de experiencias
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        # Convertir a tensores
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Predecir los valores Q actuales
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Calcular los valores Q target
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            q_target = reward_batch + GAMMA * next_q_values * (1 - done_batch)
        
        # Optimización de la red
        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Actualizar la red target
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # Reducir epsilon (exploración vs explotación)
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# Entrenamiento del agente
def train_dqn(env, agent, num_episodes):
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(200):  # Limitar a 200 pasos por episodio
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            # Castigar si el episodio termina pronto
            reward = reward if not done else -10
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.replay_experience()

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
        
        agent.decay_epsilon()

        # Actualizar la red target cada ciertos episodios
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()


# Inicializar el entorno y el agente
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

# Entrenamiento
train_dqn(env, agent, num_episodes=1000)
torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")

env.close()
