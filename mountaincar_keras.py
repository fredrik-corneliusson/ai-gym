# coding: utf-8

# https://github.com/openai/gym/wiki/MountainCar-v0
import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make('MountainCar-v0')
env.reset()

goal_steps = 200
intial_games = 1000

LEFT_PUSH = 0
NO_PUSH = 1
RIGHT_PUSH = 2

I_POSITION = 0
I_VEL = 1


class GameScore:
    def __init__(self, score, game_memory):
        self.score = score
        self.game_memory = game_memory

    @property
    def training_data(self):
        for previous_observation, action in self.game_memory:
            # Encode to one hot output array
            if action == LEFT_PUSH:
                output = [1, 0, 0]
            elif action == NO_PUSH:
                output = [0, 1, 0]
            elif action == RIGHT_PUSH:
                output = [0, 0, 1]
            yield [previous_observation, output]

    def __lt__(self, other):
        return self.score < other.score

    def __repr__(self):
        return str(self.score)


# In[5]:
POS_MIN = -1.2
POS_MAX = 0.6
VEL_MIN = -0.07
VEL_MAX = 0.07

def play_games(games=10, render=False, trained_model=None, min_score_to_return=0.50):
    scores = []
    choices = []
    top_score_percentage = 0.05
    game_scores = []
    for game_index in range(games):
        score = 0
        prev_obs = None
        game_memory = []
        for step_index in range(goal_steps):
            # if rendering, do not render every frame to speed things up.
            if render and step_index % 2 == 0:
                env.render()

            if trained_model is None \
                    or prev_obs is None:
                action = random.choice((LEFT_PUSH, NO_PUSH, RIGHT_PUSH))
            else:
                pred = trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0]
                # get the position of the maximum value
                action = np.argmax(pred)
            choices.append(action)
            new_observation, reward, done, info = env.step(action)

            # Scale values between 0-1, very important for the training
            new_observation[I_POSITION] = (new_observation[I_POSITION] - POS_MIN) / (POS_MAX - POS_MIN)
            new_observation[I_VEL] = (new_observation[I_VEL] - VEL_MIN) / (VEL_MAX - VEL_MIN)

            if prev_obs is not None:
                game_memory.append([prev_obs, action])
                # score is the max (scaled) position (1.0 is goal)
                score = max(score, new_observation[I_POSITION])

            prev_obs = new_observation
            if done:
                break
        if render:
            print(f"{game_index} {score} {step_index}")
        game_scores.append(GameScore(score, game_memory))

        env.reset()
        scores.append(score)

    avarage_score = sum(scores) / len(scores)
    print(f'Average Score: {avarage_score}', )
    print('choice LEFT:{}  choice NONE:{}  choice RIGHT:{}'.format(choices.count(LEFT_PUSH) / len(choices),
                                                                   choices.count(NO_PUSH) / len(choices),
                                                                   choices.count(RIGHT_PUSH) / len(choices)))

    top_scores = list(sorted(game_scores, reverse=True))[:(int(len(game_scores) * top_score_percentage))]

    top_scores = [top_score for top_score in top_scores if top_score.score > min_score_to_return]

    print(f"Top scores: {top_scores}")
    training_data = []
    for gs in top_scores:
        training_data.extend(gs.training_data)
    return avarage_score, training_data

def build_model(input_size, output_size):
    model = Sequential()
    # model.add(Dense(128, input_dim=input_size, activation='relu'))
    # model.add(Dense(52, activation='relu'))
    model.add(Dense(52, input_dim=input_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

def train_model(training_data, model=None):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    if not model:
        model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    epochs = 10
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

# Create the training data randomly and return average score and a portion of the best random runs to train on.
average_score, training_data = play_games(intial_games)
# Train model on the best scores
trained_model = train_model(training_data)

play_games(10, trained_model=trained_model, render=True)

# TODO: Figure out why this will cause model to degrade. Overfitting?
for i in range(4):
    average_score, training_data = play_games(50, trained_model=trained_model, render=False)
    trained_model = train_model(training_data, model=trained_model)

# play_games(10, trained_model=trained_model, render=True)

env.close()
