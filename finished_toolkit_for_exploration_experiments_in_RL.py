"""
toolkit
Usage:
python3 *.py
https://www.gymlibrary.dev/environments/classic_control/cart_pole/


add plote for ovestemation, - done
add custome size for episodes
add custome for buffer size
add list of futures
add size update off NNs
called it a day and upoad

if s_f_3=-.18 do 1 or 0, to check
this is my ocena code updates
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

print(tf. __version__) 
print(keras.__version__)
print("User Current Version:-", sys.version)
print(gym.__version__)
#print(plt.__version__)


RANDOM_SEED = 6
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))



# An episode a full game
com=input("to enter into custom setings press c or 1: ")
if com=="c" or com=="1":
    print("custtom settings menu: ")
    train_episodes = int(input("chose how many episode is whole test: "))
    ctr_ep= int(input("chose how many episode are spend in training "))
    buffer=int(input("buffer size, warning this will efect after how many action the ANN is updated: "))
    custom_deept_layers_actor= int( input("custom size of actor deppt, warning, choosing incorct dept may cause overfitting: "))
    custom_deept_layers_critic= int( input("custom size of critic deppt, warning, choosing incorct dept may cause overfitting: "))
    
else:
    print("deffult settings used ")
    train_episodes = 90
    ctr_ep=60
    buffer=2
    custom_deept_layers_actor=2
    custom_deept_layers_critic=2


def create_actor(state_shape, action_shape, func_,i=3):
    
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(124, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    for j in range(0, i):
        model.add(keras.layers.Dense(112, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
 
    model.add(keras.layers.Dense(action_shape, activation=func_ , kernel_initializer=init))
    
    model.summary()
    return model

def create_critic(state_shape, output_shape,i=4):
    
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(124, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    for j in range(0, i):
        model.add(keras.layers.Dense(112, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))

    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    
    return model

def create_actor2(state_shape, action_shape):
    
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(124, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    
    return model


#ploting fature
def plts(rewards,name_fig,r="Rewards", e="Episodes"):
	plt.xlabel(e)
	plt.ylabel(r)
	plt.title(name_fig)
	plt.plot(rewards)
	#plt.plot(ep)
	#plt.legend(loc=0)
	plt.savefig(name_fig) 
	plt.show()      


#add to check name oof off actions
def savs(fl_name,rewards_list):
    ff=open(fl_name+".txt","w")
    ff.write(str(rewards_list))
    ff.close()


def main(ctr_l):
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"
    actor_checkpoint_path2 = "training_actor/actor_cp.hdf5"
    critic_checkpoint_path2 = "training_critic/critic_cp.hdf5"


    
    actor_checkpoint_path_ = "training_actor_/actor_cp.ckpt"
    actor_checkpoint_path2_ = "training_actor_/actor_cp.hdf5"
    

    actor = create_actor(env.observation_space.shape, env.action_space.n, 'softmax', custom_deept_layers_actor)
    actor.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    actor_overest = create_actor(env.observation_space.shape, env.action_space.n, "linear" , custom_deept_layers_actor)
    actor_overest.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    critic = create_critic(env.observation_space.shape, 1 , custom_deept_layers_critic)
    critic.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    

    actor2 = create_actor2(env.observation_space.shape, env.action_space.n)
    actor2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    #add teacher ACTOR HERE




    #saaves maybe
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)

        critic.load_weights(critic_checkpoint_path)
    
    if os.path.exists('training_actor_'):
        actor2.load_weights(actor_checkpoint_path_)
        print("LOADED")
 

    #BUFFERS
    rewards_=[]
    rewards_2=[]
    obs_=[]
    advs_=[]
    acts_=[]
    long_advs_=[]
    tds_=[]
    acts_raw_=[]
    over_left_=[]
    over_rigght_=[]
    #BUFFERS

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
            # tensoers convers
            observation_reshaped = tf.convert_to_tensor(observation)
            observation_reshaped = tf.expand_dims(observation_reshaped,0)
            action_probs = actor.predict(observation_reshaped).flatten()
            action_probs2 = actor2.predict(observation_reshaped).flatten()

            
            
            #action add custom
            if ctr_l==0:
                action =(np.argmax(action_probs))
            elif ctr_l==2:
                if episode < ctr_ep:
                    if (observation[2] > .17):
                        action=1
                    else:
                        action=(np.argmax(action_probs))
                else:
                    action=(np.argmax(action_probs))
            elif ctr_l==1:
                if episode < ctr_l:
                    action = env.action_space.sample()
                else:
                    action=(np.argmax(action_probs))
            elif ctr_l==3:
                if episode < ctr_l:
                    action=(np.argmax(action_probs2))
                else:
                    action=(np.argmax(action_probs))
            else:
                print("CRASHED")
            

            next_observation, reward, done, info = env.step(action)
            next_observation_reshaped = tf.convert_to_tensor(next_observation)
            next_observation_reshaped = tf.expand_dims(next_observation_reshaped,0)
            

            #add size here for train eps_
            if (episode <ctr_ep):
                value_curr = np.asscalar(np.array(critic.predict(observation_reshaped)))
                value_next = np.asscalar(np.array(critic.predict(next_observation_reshaped)))

            # calculation of A= (r+V(s') - V (s))
                discount_factor = .7
                Temp_Diff = reward + (1 - done) * discount_factor * value_next
                advantage = Temp_Diff- value_curr
            
                advantage_reshaped = np.vstack([advantage])
                TD_target = np.vstack([Temp_Diff])

                #mid batch mormalisation traing for V*(s)
                critic.train_on_batch(observation_reshaped, TD_target)

                acts_raw_.append(action_probs)
                obs_.append(observation)
                acts_.append(action)
                advs_.append(advantage)
                tds_.append(Temp_Diff)
                
                if len(obs_)>buffer:

                    #apply gradiotion and train ANNS
                    actor.fit(np.array(obs_), np.array(acts_), sample_weight=np.array(advs_), epochs=10 ,verbose=0)
                    critic.fit(np.array(obs_), np.array(tds_), epochs=10 ,verbose=0)


                    #REAL TIME UPDATES features
                    for i in range(len(obs_)):
                        _saved_=obs_[i]
                        _saved_ = tf.convert_to_tensor(_saved_)
                        _saved_= tf.expand_dims(_saved_,0)
                        updated_=actor.predict(_saved_)
                        print("observation-f: ", np.round(obs_[i],3), " pre update acts: ", np.round(acts_raw_[i],3)," advantage used: ",np.round(advs_[i],2) , " post update acts: ", np.round(updated_[0],3))
                    
                    
                    obs_.clear()
                    acts_.clear()
                    advs_.clear()
                    long_advs_.clear()
                    tds_.clear()
                    acts_raw_.clear()
            
            observation = next_observation
            total_training_rewards += reward
            action_c+=1

            if episode > ctr_ep:
                
                #OVERESTEAMTION OBSERVATIONS FEATIRES
                over_left=(actor_overest.predict(observation_reshaped).flatten())[0]
                over_rigght=(actor_overest.predict(observation_reshaped).flatten())[1]

                over_left_.append(over_left)
                over_rigght_.append(over_rigght)

            if done:
                
                print ( "ep:",episode ,"act num:", action_c  ,"r:", np.round(reward, 2), "big r: ",total_training_rewards," adv: ",np.round(advantage, 2) ,"act:", np.round(action,2), "s-facor: ", observation, "act-space ", action )
                
                rewards_.append(total_training_rewards)
                if episode >ctr_ep:

                    actor_overest.set_weights(actor.get_weights())
                    rewards_2.append(total_training_rewards)
                    
                    
 
            


    env.close()
    if ctr_l==0:
        flag_="no restriction"
    elif ctr_l ==1:
        flag_="randomized moves"
    elif ctr_l ==2:
        flag_="restriction"
    elif ctr_l ==3:
        flag_="leaded by teacher"

    plts(rewards_,"Rewards per episode "+str(flag_))
    plts(rewards_2,"Rewards per episode_post chosing moves with "+str(flag_))

    plts(over_left_,"Overestemation left "+str(flag_),"overestemation", "actions")
    plts(over_rigght_,"Overestemation right "+str(flag_), "overestemations", "actions")
    #add plts for overestemations___

    savs("full_",rewards_)
    savs("post_tr",rewards_2)

if __name__ == '__main__':
    com=int(input("enter test rotine  \n 0 - no restriction run \n 1 - randomized exploration \n 2 - run with restrictions \n 3 - expl lead by teacher ANN : "))
    main(com)
    #main(1)
    #main(2)
    #main(3)
    #add for all tests