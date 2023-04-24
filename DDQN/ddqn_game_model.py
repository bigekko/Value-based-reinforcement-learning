import numpy as np
import os
import random
import shutil
from statistics import mean
from base_game_model import BaseGameModel
from convolutional_neural_network import ConvolutionalNeuralNetwork
from keras.models import load_model

GAMMA = 0.99
MEMORY_SIZE = 900000
BATCH_SIZE = 40
TRAINING_FREQUENCY = 100
TARGET_NETWORK_UPDATE_FREQUENCY = 300
MODEL_PERSISTENCE_UPDATE_FREQUENCY = 100





class DDQNGameModel(BaseGameModel):

    def __init__(self, game_name, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self, game_name,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)
        self.model_path = model_path
        self.ddqn = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        #print("test before here")
        if os.path.isfile(self.model_path):
         
            
            print("--------------------------------------")
            self.ddqn=load_model(self.model_path)
            print("LOADED")
        print("model path"+model_path)
        
        print("logger_path"+logger_path)
    def _save_model(self):
        self.ddqn.save(self.model_path)
        #print("saved")


class DDQNTester(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.h5"
        assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN testing",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",
                               testing_model_path)

    def move(self, state):
        if np.random.rand() < EXPLORATION_TEST:
            return random.randrange(self.action_space)
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])


class DDQNTrainer(DDQNGameModel):

    def __init__(self, game_name, input_shape, action_space):
        DDQNGameModel.__init__(self,
                               game_name,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/logs/" + game_name + "/ddqn/training/" + self._get_date() + "/",
                               "./output/neural_nets/" + game_name + "/ddqn/training/model.h5")

        if os.path.exists(os.path.dirname(self.model_path)):
            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
        os.makedirs(os.path.dirname(self.model_path))

        self.ddqn_target = ConvolutionalNeuralNetwork(self.input_shape, action_space).model
        self._reset_target_network()
       
        self.memory = []

    def move(self, state):
        
        q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
        return np.argmax(q_values[0])

    def remember(self, current_state, action, reward, next_state, terminal):
        self.memory.append({"current_state": current_state,
                            "action": action,
                            "reward": reward,
                            "next_state": next_state,
                            "terminal": terminal})
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def step_update(self, total_step):
        if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()
        if total_step % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
        
        if total_step % TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()
            self.logger.add_loss(loss)
            self.logger.add_accuracy(accuracy)
            self.logger.add_q(average_max_q)
           
       




    def _train(self):
        batch = np.asarray(random.sample(self.memory, BATCH_SIZE))
        if len(batch) < BATCH_SIZE:
            return

        current_states = []
        q_values = []
        max_q_values = []

        for entry in batch:
            current_state = np.expand_dims(np.asarray(entry["current_state"]).astype(np.float64), axis=0)
            current_states.append(current_state)
            next_state = np.expand_dims(np.asarray(entry["next_state"]).astype(np.float64), axis=0)
            next_state_prediction = self.ddqn_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_prediction)
            q = list(self.ddqn.predict(current_state)[0])
            if entry["terminal"]:
                q[entry["action"]] = entry["reward"]
            else:
                q[entry["action"]] = entry["reward"] + GAMMA * next_q_value
            q_values.append(q)
            max_q_values.append(np.max(q))

        fit = self.ddqn.fit(np.asarray(current_states).squeeze(),
                            np.asarray(q_values).squeeze(),
                            batch_size=BATCH_SIZE,
                            verbose=0)
        loss = fit.history["loss"][0]
        accuracy = fit.history["acc"][0]
        return loss, accuracy, mean(max_q_values)

 
    def _reset_target_network(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())