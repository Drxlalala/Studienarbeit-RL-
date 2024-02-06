This incloud the train code, training result, code for text, abandoned code.
The training codes are: 

“Interation.py” After each training get the parameters of the model and set the model parameters for next training with the average of the top five. But It has little impact on training and is inconsistent, so it is abandoned.

"sb3_train_play_underwater.py" and "sb3_train_play.py" are for text the Env and to show the training result.

"SAC", "DDPG", "ulits" are the Rl algorithm wirte by myself. But in training, the model can not with these two algorithm convergence, so they were abandoned. Then use "stable_baselines3".

Under "Training code":
"curriculum_level1.py" is train the arm in "Normal Env" to reach the object
"curriculum_level2.py" is train the arm in "Normal Env" to take and place the object
"curriculum_level3.py" is train the arm in "Underwater Env" to take and place the object
"underwater_hybrid.py" is train the arm in "Underwater Env" with hybrid reward function
"underwater_sparse.py" is train the arm in "Underwater Env" with sparse reward function

Under "Result":
It shows the 6 training results. 
