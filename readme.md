# Readme
> Many environment interaction settings are acquired from [DQN_HollowKnight](https://github.com/ailec0623/DQN_HollowKnight). Many Thanks to its author!

> I have rebased the structrue of the whole trainning procedure, mentioned by deleting the mannually designed action pairs, designing a contrastive representation learning for RL and so on. 

> In the future, I will try model-based RL methods, or even imitation learning from human playing experience...

## Environment

- windows 10 (We use win32 API to operate the little knight and get screenshots)
- python 3.8.8
- python liberary: find in `requirments.txt`
- Hollow Knight
- HP Bar mod for Hollow Knight (In order to get the boss hp to calculate the reward, please find the mod in `./hollow_knight_Data/`, and then copy the mod file to the game folder)
- CUDA and cudnn for tensorflow

## Usage

- Now I only write train.py but not test.py (the file is just test some base functions not for model), you can write it by yourself if you get a good model.
- I upload a saving file, if you never played this game, please move `/save_file/user3.dat` into save folder (usually `C:\user\_username_\AppData\LocalLow\Team Cherry\Hollow Knight`)
- Adjust the game resolution to 1920*1017 
- Run train.py
- Keep the game window at the forefront (Since I cannot send keyboard event in the background, I tried `PossMassage()` in win32 API, but it did not work well.
                                         If you have any idea about sending keyboard event in the background, please let me know)
- Let the little knight stand in front of the statue of the boss in the godhome
- Press `F1` to start trainning. (Also you can use `F1` to stop trainning)

