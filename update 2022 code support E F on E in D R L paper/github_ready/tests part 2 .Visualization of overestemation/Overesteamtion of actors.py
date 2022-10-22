"""
A minimal Advantage Actor Critic Implementation
Usage:
python3 *.py
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
train_episodes = 120

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    #model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    #model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def plts(rewards,name_fig):
	plt.xlabel("Episodes")
	plt.ylabel("Rewards")
	plt.title(name_fig)
	plt.plot(rewards)
	#plt.plot(ep)
	#plt.legend(loc=0)
	plt.savefig(name_fig) 
	plt.show()      

def plts2(rewards,name_fig):
	plt.xlabel("Episodes")
	plt.ylabel("Overestemation")
	plt.title(name_fig)
	plt.plot(rewards)
	#plt.plot(ep)
	#plt.legend(loc=0)
	plt.savefig(name_fig) 
	plt.show() 




def savs(fl_name,rewards_list):
    ff=open(fl_name+".txt","w")
    ff.write(str(rewards_list))
    ff.close()


def main():
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"
    actor_checkpoint_path2 = "training_actor/actor_cp.hdf5"
    critic_checkpoint_path2 = "training_critic/critic_cp.hdf5"

    actor = create_actor(env.observation_space.shape, env.action_space.n)
    actor.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    critic = create_critic(env.observation_space.shape, 1)
    critic.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)

        critic.load_weights(critic_checkpoint_path)
        print("sucsefully loaded testing ANN")
    else:
        print("No ANN loaded to test, unvalid")
        sys.exit()

    #print(actor)
    #print(critic)

    # X = states, y = actions
    X = []
    y = []
    Left_=[]
    Righ_=[]
    rewards_=[]
    rewards_2=[]
    for episode in range(train_episodes):
        total_training_rewards = 0
        action_c=0
        observation = env.reset()
        done = False
        while not done:
            if True:
                env.render()
                pass

            # model dims are (batch, env.observation_space.n)
            #observation_reshaped = observation.reshape([1, observation.shape[0]])
            observation_reshaped = tf.convert_to_tensor(observation)
            observation_reshaped = tf.expand_dims(observation_reshaped,0)
            action_probs = actor.predict(observation_reshaped).flatten()

            action =(np.argmax(action_probs))
            print(action_probs[0])
            Left_.append(action_probs[0])
            print(action_probs[1])
            Righ_.append(action_probs[1])
            encoded_action = one_hot_encode_action(action, env.action_space.n)

            next_observation, reward, done, info = env.step(action)
            next_observation_reshaped = tf.convert_to_tensor(next_observation)
            next_observation_reshaped = tf.expand_dims(next_observation_reshaped,0)
            #next_observation_reshaped = next_observation.reshape([1, next_observation.shape[0]])

            if (episode <0):
                value_curr = np.asscalar(np.array(critic.predict(observation_reshaped)))
                value_next = np.asscalar(np.array(critic.predict(next_observation_reshaped)))

            # Fit on the current observation
                discount_factor = .7
                TD_target = reward + (1 - done) * discount_factor * value_next
                advantage = critic_target = TD_target - value_curr
            #print(np.around(action_probs, 2), np.around(value_next - value_curr, 3), 'Advantage:', np.around(advantage, 2))
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
            action_c+=1
            if done:
                print('Total training rewards: {} after n steps = {} with final reward = {}'.format(total_training_rewards, episode, reward))
                print ( "ep:",episode ,"act num:", action_c  ,"r:", np.round(reward, 2), "big r: ",total_training_rewards ,"act:", np.round(action,2) )
                #total_training_rewards += 1
                rewards_.append(total_training_rewards)
                if episode >0:
                    rewards_2.append(total_training_rewards)
                    if os.path.exists('training_actor'):
                        print("no update")
                    else:
                        actor.save_weights(actor_checkpoint_path)
                        actor.save_weights(actor_checkpoint_path2)
                        critic.save_weights(critic_checkpoint_path)
                        critic.save_weights(critic_checkpoint_path2)
            if total_training_rewards > 60:

                print('very cool')

    env.close()
    plts(rewards_,"Rewards per episode")
    plts(rewards_2,"Rewards per episode_post 60")
    plts2(Left_,"overestemation Left")
    plts2(Righ_,"overestemation Right")
    savs("full_",rewards_)
    savs("post_tr",rewards_2)
    savs("Left_Q",Left_)
    savs("Right_Q",Righ_)
if __name__ == '__main__':
    main()
