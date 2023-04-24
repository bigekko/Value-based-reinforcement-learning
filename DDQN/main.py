import gym
import numpy as np
from ddqn_game_model import DDQNTrainer, DDQNTester

from gym_wrappers import MainGymWrapper

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Atari:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_run_limit, clip = self.gameargs()
        env_name = game_name + "Deterministic-v4" 
        env = MainGymWrapper.wrap(gym.make(env_name))
        self.main_loop(self.game_models(game_mode, game_name, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)

    def main_loop(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        

        run = 0
        total_step = 0
        while True:
            if total_run_limit is not None and run >= total_run_limit:
                print("Reached total run limit of: " + str(total_run_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print( "Reached total step limit of: " + str(total_step_limit))
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                #print( "State: " , next_state)
                if clip:
                    np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                game_model.step_update(total_step)
                
                if terminal:
                    game_model.save_run(score, step, run)
                    break
            #print(run)

    def gameargs(self):
    
        game_name="SpaceInvaders"
        game_mode="ddqn_training"
        render=True
        total_step_limit=1000000
        total_run_limit=5
        clip=False #normalize/clip reward to 0-1
        print("Game name: " + str(game_name))
        print("Selected mode: " + str(game_mode))
        print("Should render: " + str(render))
        print("Should normalize reward: " + str(clip))
        print("Total step limit: " + str(total_step_limit))
        print("Total run limit: " + str(total_run_limit))
        return game_name, game_mode, render, total_step_limit, total_run_limit, clip

    def game_models(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNTester(game_name, INPUT_SHAPE, action_space)
        
            print("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Atari()
