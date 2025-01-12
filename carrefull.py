#
#   Version 1.0
#   January 2025
#
#   Developed by Francesco Zazza
#   francescozazza@gmail.com
#
#   Have a nice fun! :-)
#

import pygame
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

# Inizializza Pygame
pygame.init()

# Costanti di Gioco
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GRID_SIZE = 8
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# Colori
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)

# Carichiamo le immagini
cliente_normale_image = pygame.image.load("assets/cliente_normale.png")
cliente_banana_image  = pygame.image.load("assets/cliente_banana.png")
banana_image          = pygame.image.load("assets/banana.png")
start_image           = pygame.image.load("assets/startflag.png")
explosion_image       = pygame.image.load("assets/explosion.png")   # Per il competitor
party_image           = pygame.image.load("assets/party.png")       # <--- Nuova immagine per Carrefull

# Carichiamo i supermercati
supermarket_images = {
    "Esselenta": pygame.image.load("assets/esselenta.png"),
    "RidL":      pygame.image.load("assets/ridl.png"),
    "Carrefull": pygame.image.load("assets/carrefull.png"),
    "Spam":      pygame.image.load("assets/spam.png"),
}

class CarrefullEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        
        # Il cliente parte da (0, grid_size/2)
        self.client_pos = [0, self.grid_size/2]

        # Banane random: al massimo 2
        # per _ in range(2): genera 2 banane
        self.bananas = [
            (random.randint(0, self.grid_size - 1), 
             random.randint(1, self.grid_size - 2))
            for _ in range(2)  # <--- max 2 banane
        ]

        # Definiamo la copertura (lista di celle) di ogni supermercato 2x2.
        self.supermarkets_coverage = {
            "Esselenta": [(0,0), (1,0), (0,1), (1,1)],
            "RidL":      [(0,6), (1,6), (0,7), (1,7)],
            "Carrefull": [(6,0), (7,0), (6,1), (7,1)],
            "Spam":      [(6,6), (7,6), (7,7), (7,7)],
        }
        
        self.on_banana = False

    def reset(self):
        # Riposiziona il cliente, azzera la banana
        self.client_pos = [0, self.grid_size/2]
        self.on_banana = False

        # rifacciamo il "Carrefull_pos" come prima
        carrefull_pos = [10, 0]
        return self.client_pos + carrefull_pos

    def step(self, action):
        """
        Azioni:
        0 -> Su
        1 -> Giù
        2 -> Sinistra
        3 -> Destra
        """
        x, y = self.client_pos
        
        if action == 0:   # up
            y = max(0, y - 1)
        elif action == 1: # down
            y = min(self.grid_size - 1, y + 1)
        elif action == 2: # left
            x = max(0, x - 1)
        elif action == 3: # right
            x = min(self.grid_size - 1, x + 1)
        
        self.client_pos = [x, y]
        carrefull_pos = [10, 0]  # invariato

        # Ricompensa e done
        # 1) Carrefull
        if tuple(self.client_pos) in self.supermarkets_coverage["Carrefull"]:
            print("\033[92m ✅ Carrefull!!! Scelta giusta!!!\033[0m")
            return self.client_pos + carrefull_pos, 40, True
        
        # 2) Banana
        if tuple(self.client_pos) in self.bananas:
            self.on_banana = True
            print("\033[93m 🍌 Banana\033[0m")
            return self.client_pos + carrefull_pos, -1, False
        
        # 3) Competitor
        for name, coverage in self.supermarkets_coverage.items():
            if name == "Carrefull":
                continue
            if tuple(self.client_pos) in coverage:
                print("\033[91m ❌ Competitor: {}\033[0m".format(name))
                return self.client_pos + carrefull_pos, -40, True
        
        # 4) Passo neutro
        self.on_banana = False
        return self.client_pos + carrefull_pos, -1, False


# Funzioni di disegno
def draw_grid(screen):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y))

def draw_start(screen):
    x, y = [0, 3]
    image = start_image
    start_scaled = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
    screen.blit(start_scaled, (x * CELL_SIZE, y * CELL_SIZE))

def draw_client(screen, client_pos, on_banana):
    x, y = client_pos
    image = cliente_banana_image if on_banana else cliente_normale_image
    cliente_scaled = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
    screen.blit(cliente_scaled, (x * CELL_SIZE, y * CELL_SIZE))

def draw_bananas(screen, bananas):
    for banana in bananas:
        x, y = banana
        banana_scaled = pygame.transform.scale(banana_image, (CELL_SIZE, CELL_SIZE))
        screen.blit(banana_scaled, (x * CELL_SIZE, y * CELL_SIZE))

def draw_supermarkets(screen, supermarkets_coverage):
    for name, coverage in supermarkets_coverage.items():
        min_x = min(pos[0] for pos in coverage)
        min_y = min(pos[1] for pos in coverage)
        image = supermarket_images[name]
        supermarket_scaled = pygame.transform.scale(image, (CELL_SIZE * 2, CELL_SIZE * 2))
        screen.blit(supermarket_scaled, (min_x * CELL_SIZE, min_y * CELL_SIZE))

# Agente DQN
class DQNAgent:
    def __init__(self, grid_size, gamma=0.99, learning_rate=0.01, memory_size=2000, batch_size=64):
        self.grid_size = grid_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def predict(self, state):
        state = np.array(state).reshape(1, -1)
        return self.model.predict(state, verbose=0)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        self.model.fit(states, q_values, verbose=0, batch_size=self.batch_size)


# Ciclo principale
if __name__ == "__main__":
    env = CarrefullEnv()
    agent = DQNAgent(grid_size=GRID_SIZE)

    epsilon = 0.9
    epsilon_decay = 0.950
    epsilon_min = 0.1

    rewards_history = []

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("DQN - Carrefull Environment")

    num_episodes = 500
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            # Rendering
            screen.fill(WHITE)
            draw_grid(screen)
            draw_supermarkets(screen, env.supermarkets_coverage)
            draw_bananas(screen, env.bananas)
            draw_start(screen)
            draw_client(screen, env.client_pos, env.on_banana)
            pygame.display.flip()

            # Epsilon-greedy
            if np.random.rand() < epsilon:
                action = random.choice([0, 1, 2, 3])
                print(f"Epsilon {epsilon:.4f} - Episode {episode + 1}: Esplorazione")
            else:
                q_values = agent.predict(state)
                action = np.argmax(q_values)
                print(f"Epsilon {epsilon:.4f} - Episode {episode + 1}: Q-Values: {q_values}")

            # Esegui step
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            agent.train()

            state = next_state
            total_reward += reward

            # Se finisce su Competitor (reward = -40)
            if done and reward == -40:
                # Disegno l'explosion
                x, y = env.client_pos
                explosion_scaled = pygame.transform.scale(explosion_image, (CELL_SIZE, CELL_SIZE))
                screen.blit(explosion_scaled, (x * CELL_SIZE, y * CELL_SIZE))
                pygame.display.flip()
                pygame.time.wait(500)  # 0,5 secondi
                break

            # Se finisce su Carrefull (reward = 40)
            if done and reward == 40:
                # Disegno la festa "party"
                x, y = env.client_pos
                party_scaled = pygame.transform.scale(party_image, (CELL_SIZE, CELL_SIZE))
                screen.blit(party_scaled, (x * CELL_SIZE, y * CELL_SIZE))
                pygame.display.flip()
                pygame.time.wait(500)  # 0,5 secondi
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        print("###########################")
        print(f"Episodio {episode + 1}: Ricompensa = {total_reward}, Epsilon = {epsilon}")
        print("###########################")

    # Fine training: salviamo il modello
    agent.model.save("dqn_carrefull.h5")
    print("Modello salvato come 'dqn_carrefull.h5'.")

    # Grafico delle ricompense
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Ricompensa per episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Ricompensa')
    plt.title("Andamento delle ricompense durante l'allenamento")
    plt.legend()
    plt.show()
