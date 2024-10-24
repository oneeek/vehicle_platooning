#%%
import Pendulum



# train
HC = Pendulum.Pendulum()
max_episode = 100
Reward = HC.train(max_episode)
file_path_a, file_path_c = HC.save_weights()

#%% 

# test
HC = Pendulum.Pendulum()
HC.load_weights(file_path_a, file_path_c)
HC.test()

#%%
