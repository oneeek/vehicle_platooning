#%%
import HalfCheetah



# train
HC = HalfCheetah.HalfCheetah()
max_episode = 200
Reward = HC.train(max_episode)
file_path_a, file_path_c = HC.save_weights()

#%% 

# test
HC = HalfCheetah.HalfCheetah()
HC.load_weights(file_path_a, file_path_c)
HC.test()

#%%
