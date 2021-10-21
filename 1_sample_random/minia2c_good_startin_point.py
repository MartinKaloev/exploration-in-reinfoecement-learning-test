"""
A minimal Advantage Actor Critic Implementation
Usage:
python3 minA2C.py
"""
import sys
import gym
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
import time
import random

RANDOM_SEED = 6
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
#env = gym.make('MountainCar-v0')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def plts(rewards,name_fig, Scales):
	plt.xlabel("Episodes")
	plt.ylabel(Scales)
	plt.title(name_fig)
	plt.plot(rewards)
	#plt.plot(ep)
	#plt.legend(loc=0)
	plt.savefig(name_fig) 
	plt.show()      



def main():
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"

    actor = create_actor(env.observation_space.shape, env.action_space.n)
    critic = create_critic(env.observation_space.shape, 1)
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)

        critic.load_weights(critic_checkpoint_path)
    #print(actor)
    #print(critic)

    # X = states, y = actions
    X = []
    y = []
    rewards_=[]
    OvQ_R =[]
    OvQ_L =[]
    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            if True:
                #env.render()
                pass

            # model dims are (batch, env.observation_space.n)
            observation_reshaped = observation.reshape([1, observation.shape[0]])
            action_probs = actor.predict(observation_reshaped).flatten()
            # Note we're sampling from the prob distribution instead of using argmax
            #action = np.random.choice(env.action_space.n, 1, p=action_probs)[0]
            
            if episode <60:
                action = env.action_space.sample()
            else:
                #action = ((np.around(action_probs, 2)[0], np.around(action_probs, 2)[1]))
                action = (np.argmax(action_probs))
                action2 = action_probs [action]
                
            #print ("action is " ,action)
            encoded_action = one_hot_encode_action(action, env.action_space.n)
            #print ("encoded is ", encoded_action)
            next_observation, reward, done, info = env.step(action)
            next_observation_reshaped = next_observation.reshape([1, observation.shape[0]])

            value_curr = np.asscalar(np.array(critic.predict(observation_reshaped)))
            value_next = np.asscalar(np.array(critic.predict(next_observation_reshaped)))

            # Fit on the current observation
            discount_factor = .7
            TD_target = reward + (1 - done) * discount_factor * value_next
            advantage = critic_target = TD_target - value_curr
            OvQ_L.append(np.around(action_probs, 2)[0])
            OvQ_R.append(np.around(action_probs, 2)[1])
            print("Observations positio: ", np.around(next_observation, 2)[0],"Observations velocity: ",  np.around(next_observation, 2)[1],"Observations angle: ",  np.around(next_observation, 2)[2],"Observations Pole Angular Velocity : ",  np.around(next_observation, 2)[3])
            #print(np.around(action_probs, 2)[0], np.around(action_probs, 2)[1], np.around(value_next - value_curr, 3), 'Advantage:', np.around(advantage, 2))
            advantage_reshaped = np.vstack([advantage])
            TD_target = np.vstack([TD_target])
            critic.train_on_batch(observation_reshaped, TD_target)
            #critic.fit(observation_reshaped, TD_target, verbose=0)

            gradient = encoded_action - action_probs
            gradient_with_advantage = .0001 * gradient * advantage_reshaped + action_probs
            actor.train_on_batch(observation_reshaped, gradient_with_advantage)
            #actor.fit(observation_reshaped, gradient_with_advantage, verbose=0)
            observation = next_observation
            total_training_rewards += reward
            rewards_.append(total_training_rewards)
            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                total_training_rewards += 1
                
            if total_training_rewards > 90 or episode == 160:
                actor.save_weights(actor_checkpoint_path)
                critic.save_weights(critic_checkpoint_path)
                plts(rewards_,"Rewards per episode", "Rewards")
                plts(OvQ_R,"Overestimation Right", "Overestimation") 
                plts(OvQ_L,"Overestimation Left", "Overestimation")  
                sys.exit()

    env.close()

if __name__ == '__main__':
    main()
