import random
import numpy as np
import tensorflow as tf
from collections import namedtuple
from statistics import mean
import pandas as pd
from replay_memory import ReplayMemory
from agent import Agent
from datetime import datetime
from tensorflow.keras.layers import Dense, Flatten

DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent(Agent):   

    def __init__(self,
                 env: 'TradingEnv',
                 policy_network: tf.keras.Model = None, 
		 window_size_from_env = None):
        self.env = env
        self.n_actions = env.action_space
        print(self.n_actions)
        self.observation_shape = env.observation_space.shape
        print(self.observation_shape)
        self.window_steps_per_episode = 64 #trace_length, tamanho de cada episodio 128 É UM BOM START
        self.policy_network = policy_network or self._build_policy_network(self.window_steps_per_episode)
        #self.policy_network = MyModel(self.observation_shape, self.n_actions, self.window_steps_per_episode)

        self.target_network = policy_network or self._build_policy_network(self.window_steps_per_episode)
        #self.target_network = self._build_policy_network(self.window_steps_per_episode)
        #self.target_network = MyModel(self.observation_shape, self.n_actions, self.window_steps_per_episode)
        #self.target_network.trainable = False
        self.id = 'TraderGOD'
        self.env.agent_id = self.id
        #self.network_weights_for_predict = self._build_policy_network(1)
        self.window_size_from_env = window_size_from_env
        

    #def _build_policy_network(self):
    #    network = tf.keras.Sequential([
    #        tf.keras.layers.InputLayer(input_shape=self.observation_shape),
    #        tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
    #        tf.keras.layers.MaxPooling1D(pool_size=2),
    #        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"),
    #        tf.keras.layers.MaxPooling1D(pool_size=2),
    #        tf.keras.layers.Flatten(),
    #        tf.keras.layers.Dense(self.n_actions, activation="sigmoid"),
    #        tf.keras.layers.Dense(self.n_actions, activation="softmax")
    #    ])
    #
    #    return network

    def _build_policy_network_old(self):
        model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=self.observation_shape),
        #tf.keras.layers.Conv1D(filters=128, kernel_size=6, padding="same", activation='relu'),
        #tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding="same", activation='relu'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding="same", activation='relu'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding="same", activation='relu'),
       
        #tf.keras.layers.Flatten(),
        
        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
        
        # Use last trace for training
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(self.n_actions, activation='linear')        
        ])
        
        return model

    def _build_policy_network(self, batch_size):
        batch_size = batch_size
        #kernel_size relacionado às features X time
        #https://www.macnica.co.jp/business/ai_iot/columns/135112/
        #https://medium.com/@jon.froiland/convolutional-neural-networks-for-sequence-processing-part-1-420dd9b500
        #https://stackoverflow.com/questions/51344610/how-to-setup-1d-convolution-and-lstm-in-keras
        inputs = tf.keras.Input(shape=self.observation_shape)
        print(inputs)
        a = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', kernel_initializer='HeNormal')(inputs)
        print(a)
        #a = tf.keras.layers.MaxPooling1D(pool_size=2)(a)
        #print(a)
        a = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu', kernel_initializer='HeNormal')(a)
        #print(a)
        #a = tf.keras.layers.MaxPooling1D(pool_size=2)(a)
        #print(a)
        #a = tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', kernel_initializer='HeNormal')(a)
        #print(a)
        #cell = tf.keras.layers.LSTMCell(256)
        #self.lstm = tf.keras.layers.RNN(cell, stateful = True)(a)
        lstm  = tf.keras.layers.LSTM(64, return_sequences=True)(a)
        lstm_1  = tf.keras.layers.LSTM(32)(lstm)       
        #lstm_2 = tf.keras.layers.LSTM(128)(lstm_1)
        #lstm  = tf.keras.layers.LSTM(64)(a)
        #flat = tf.keras.layers.Flatten()(lstm)
        
        #print(lstm)
        #ult_dense = tf.keras.layers.Dense(32, activation='elu')(lstm_1)
        #ult_dense_1 = tf.keras.layers.Dense(16, activation='relu')(ult_dense)

        outputs = tf.keras.layers.Dense(self.n_actions.n, activation='linear')(lstm_1)

        #model = tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=self.observation_shape),
        ##tf.keras.layers.Conv1D(filters=128, kernel_size=6, padding="same", activation='relu'),
        ##tf.keras.layers.MaxPooling1D(pool_size=2),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'),
       
        #tf.keras.layers.Flatten(),
        
        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
        
        # Use last trace for training
        #tf.keras.layers.LSTM(64,  activation='relu'),
        #tf.keras.layers.Dense(self.n_actions, activation='linear')        
        #])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def _build_policy_network___REDE_MINHA(self, batch_size):
        batch_size = batch_size
        #kernel_size relacionado às features X time
        #https://www.macnica.co.jp/business/ai_iot/columns/135112/
        #https://medium.com/@jon.froiland/convolutional-neural-networks-for-sequence-processing-part-1-420dd9b500
        #https://stackoverflow.com/questions/51344610/how-to-setup-1d-convolution-and-lstm-in-keras
        inputs = tf.keras.Input(shape=self.observation_shape)
        print(inputs)
        a = tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu', kernel_initializer='HeNormal')(inputs)
        print(a)
        #a = tf.keras.layers.MaxPooling1D(pool_size=2)(a)
        #print(a)
        a = tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer='HeNormal')(a)
        #print(a)
        #a = tf.keras.layers.MaxPooling1D(pool_size=2)(a)
        #print(a)
        a = tf.keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', kernel_initializer='HeNormal')(a)
        #print(a)
        #cell = tf.keras.layers.LSTMCell(256)
        #self.lstm = tf.keras.layers.RNN(cell, stateful = True)(a)
        lstm  = tf.keras.layers.LSTM(512, return_sequences=True)(a)
        lstm_1  = tf.keras.layers.LSTM(256, return_sequences=True)(lstm)       
        lstm_2 = tf.keras.layers.LSTM(128)(lstm_1)
        #lstm  = tf.keras.layers.LSTM(64)(a)
        
        #print(lstm)
        ult_dense = tf.keras.layers.Dense(64, activation='relu')(lstm_2)
        #ult_dense_1 = tf.keras.layers.Dense(16, activation='relu')(ult_dense)

        outputs = tf.keras.layers.Dense(self.n_actions.n, activation='linear')(ult_dense)

        #model = tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=self.observation_shape),
        ##tf.keras.layers.Conv1D(filters=128, kernel_size=6, padding="same", activation='relu'),
        ##tf.keras.layers.MaxPooling1D(pool_size=2),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'),
       
        #tf.keras.layers.Flatten(),
        
        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
        
        # Use last trace for training
        #tf.keras.layers.LSTM(64,  activation='relu'),
        #tf.keras.layers.Dense(self.n_actions, activation='linear')        
        #])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def _build_policy_network_estranha(self, batch_size):
        batch_size = batch_size
        #https://www.naun.org/main/NAUN/neural/2018/a162016-062.pdf
        inputs = tf.keras.Input(shape=self.observation_shape)
        print(inputs)
        a = tf.keras.layers.Dense(128, activation='relu')(inputs)
        print(a)
        b = tf.keras.layers.Conv1D(filters=24, kernel_size=10, activation='relu')(a)
        print(b)
        c = tf.keras.layers.MaxPooling1D(pool_size=2)(b)
        print(c)
        d = tf.keras.layers.Conv1D(filters=48, kernel_size=5,  activation='relu')(c)
        print(d)
        e = tf.keras.layers.MaxPooling1D(pool_size=2)(d)
        print(e)
        #a = tf.keras.layers.Conv1D(filters=32, kernel_size=15, activation='relu')(a)
        #print(a)
        #cell = tf.keras.layers.LSTMCell(256)
        #self.lstm = tf.keras.layers.RNN(cell, stateful = True)(a)

        lstm1  = tf.keras.layers.LSTM(40, return_sequences=True)(e)
        print(lstm1)       
        lstm  = tf.keras.layers.LSTM(32)(lstm1)
        #print(lstm)
        ult_dense = tf.keras.layers.Dense(32, activation='relu')(lstm)
        print((ult_dense))
        

        outputs = tf.keras.layers.Dense(self.n_actions.n, activation='linear')(ult_dense)

        #model = tf.keras.Sequential([
        #tf.keras.layers.InputLayer(input_shape=self.observation_shape),
        ##tf.keras.layers.Conv1D(filters=128, kernel_size=6, padding="same", activation='relu'),
        ##tf.keras.layers.MaxPooling1D(pool_size=2),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=8, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding="same", activation='relu'),
        #tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation='relu'),
       
        #tf.keras.layers.Flatten(),
        
        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
        
        # Use last trace for training
        #tf.keras.layers.LSTM(64,  activation='relu'),
        #tf.keras.layers.Dense(self.n_actions, activation='linear')        
        #])
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
    
    def _build_policy_network_melhor(self, batch_size):
        batch_size = batch_size
        #https://www.naun.org/main/NAUN/neural/2018/a162016-062.pdf
        inputs = tf.keras.Input(shape=self.observation_shape)
        print(inputs)        
        
        a = tf.keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu',padding='same', kernel_initializer='HeNormal')(inputs)
        print(a)
        
        print(a)
        a = tf.keras.layers.Conv1D(filters=256, kernel_size=5, activation='relu',padding='same', kernel_initializer='HeNormal')(a)
        print(a)
        
        a = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu',padding='same', kernel_initializer='HeNormal')(a)
        print(a)
        #a = tf.keras.layers.Conv1D(filters=32, kernel_size=15, activation='relu')(a)
        #print(a)
        #cell = tf.keras.layers.LSTMCell(256)
        #self.lstm = tf.keras.layers.RNN(cell, stateful = True)(a)
        #a = tf.keras.layers.GlobalAveragePooling1D()(a)
        #print(a)
        lstm  = tf.keras.layers.LSTM(64, return_sequences=True)(a)        
        print(lstm)
        max_pol = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        ult_dense = tf.keras.layers.Dense(32, activation='softmax')(max_pol)
        print((ult_dense))
        

        outputs = tf.keras.layers.Dense(self.n_actions.n, activation='linear')(ult_dense)

        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
        

    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.policy_network.save(path + filename)
        return filename

    def get_action(self, state: np.ndarray, threshold ) -> int:
        #threshold: float = kwargs.get('threshold', 0)

        rand = random.random()

        if rand < threshold:
            return np.random.choice(self.n_actions.n)
        else:
            #print("ENTRADA", self.observation_shape)
            #self.policy_network.get_layer(name = 'lstm').reset_states(states=None)
            #print(state)
            #print(state.shape)
            #print("HAKUNA MATATA", np.expand_dims(state, 0))
            #print(np.expand_dims(state, 0).shape)
            #print("PUMBA",self.policy_network(np.expand_dims(state, 0)) )
            #self.network_weights_for_predict.set_weights(self.policy_network.get_weights())
            #state_tensor = tf.expand_dims(state, 0)
            return tf.argmax(self.policy_network(tf.expand_dims(state, 0))[0]).numpy()
            #return np.argmax(self.network_weights_for_predict(np.expand_dims(state, 0)))
            #return np.argmax(self.policy_network((np.atleast_2d(state))[0]))

    def _apply_gradient_descent_DQN(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor: float, trace_length):
        
        #clipnorm
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=1.0)
        loss = tf.keras.losses.Huber()
        erro = list()

        #min_history = trace_length*1.5
        #if len(memory) < min_history:
        #    return 0
        i = 0
        while i < batch_size:
            #print("ANTES DE FORWARD", self.lstm_out_c, self.lstm_out_state)
            #trace_length -> tamanho da janela q vai ter cada ep do batch
            transitions = memory.sample_DRQN_pos_DODO_gambia(trace_length = trace_length )
            #print("NAO SEI MAIS", transitions)
            #print("NAO SEI MAIS MUDO", transitions)
            batch = DQNTransition(*zip(*transitions))
            #print("NAO SEI MAIS", batch)
            #variables = self.policy_network.trainable_variables

            #with tf.GradientTape() as tape:
            #tape.watch(variables)
                #print(batch.state)
            state_batch = tf.convert_to_tensor(batch.state)
                #print(state_batch)
            action_batch = tf.convert_to_tensor(batch.action)
            reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(batch.next_state)
                #print(next_state_batch.shape)
            done_batch = tf.convert_to_tensor(batch.done, dtype=tf.float32)
                #print(done_batch)
            target_q = self.target_network(next_state_batch)
                #next_action = np.argmax(target_q, axis=1)
            next_action = tf.reduce_max(target_q, axis=1)
                #target_value = tf.reduce_sum(tf.one_hot(next_action, self.n_actions.n) * target_q, axis=1)
            actual_values = (1-done_batch) *  discount_factor * next_action + reward_batch

                #target_q = self.target_network(next_state_batch)
                #next_action = tf.argmax(target_q, axis=1)
                #target_value = tf.reduce_sum(tf.one_hot(next_action, self.n_actions.n) * target_q, axis=1)
                #actual_values = (1-done_batch) *  discount_factor * target_value + reward_batch

                #value_next = np.max(self.target_network(next_state_batch), axis=1)
                #actual_values = np.where(done_batch, reward_batch, reward_batch + discount_factor*value_next)
                #real values s the target value that the network attempts to get closer to on each iteration.

                #

                #state_action_values = tf.math.reduce_sum(tf.multiply(
                #    self.policy_network(state_batch),tf.one_hot(action_batch, self.n_actions.n)),
                #    axis=1
                #)
            with tf.GradientTape() as tape:    
                #state_action_values = tf.math.reduce_sum(
                #    self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions.n),
                #    axis=1
                #)
                state_action_values = tf.reduce_sum(tf.multiply(
                    self.policy_network(state_batch),tf.one_hot(action_batch, self.n_actions.n)),
                    axis=1
                )
                #next_state_values = tf.where(
                #    done_batch,
                #    tf.zeros(batch_size*trace_length), #REPLAY COM O DRQN, ADD O *TRACE_LENGTH
                #    tf.math.reduce_max(self.target_network(next_state_batch), axis=1)
                #)

                #expected_state_action_values = reward_batch + (discount_factor * next_state_values)
                loss_value = loss(actual_values, state_action_values)
            
            #print(loss_value)
        #print(state_action_values)
        #print(actual_values)
            #variables = self.policy_network.trainable_variables
        #print(variables)
            gradients = tape.gradient(loss_value,self.policy_network.trainable_variables)
            #erro.append(loss_value)            
        #print(gradients)
            optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
            if isinstance(loss_value, int):
                erro.append(loss_value)
            else:
                erro.append(loss_value.numpy())
            i += 1
            #print("DEPOIS DO BACK DE CADA EP", tf.print(self.lstm_out_c,output_stream=sys.stderr), tf.print(self.lstm_out_state,output_stream=sys.stderr))
        
        
        #//self.policy_network.get_layer(name = 'lstm').reset_states(states=None) 

        return mean(erro)

    def _apply_gradient_descent_MAISCONSISTENTE(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor, trace_length):
        
        #clipnorm
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate, clipvalue=1.0, clipnorm = 1.0)
        loss = tf.keras.losses.Huber()
        erro = list()
        
        #min_history = trace_length*1.5
        #if len(memory) < min_history:
        #    return 0
        i = 0
        while i < batch_size:
            #print("ANTES DE FORWARD", self.lstm_out_c, self.lstm_out_state)
            #trace_length -> tamanho da janela q vai ter cada ep do batch
            transitions = memory.sample_DRQN_pos_DODO_gambia(trace_length = trace_length )
            #print("NAO SEI MAIS", transitions)
            #print("NAO SEI MAIS MUDO", transitions)
            batch = DQNTransition(*zip(*transitions))
            #print("NAO SEI MAIS", batch)
            variables = self.policy_network.trainable_variables

            with tf.GradientTape() as tape:
                tape.watch(variables)

                state_batch = tf.convert_to_tensor(batch.state)
                #print(state_batch)
                action_batch = tf.convert_to_tensor(batch.action)
                reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
                next_state_batch = tf.convert_to_tensor(batch.next_state)
                #print(next_state_batch.shape)
                done_batch = tf.convert_to_tensor(batch.done, dtype=tf.float32)
                #print(done_batch)
                target_q = self.target_network(next_state_batch)
                next_action = tf.argmax(target_q, axis=1)
                target_value = tf.reduce_sum(tf.one_hot(next_action, self.n_actions.n) * target_q, axis=1)
                actual_values = (1-done_batch) *  discount_factor * target_value + reward_batch

                #value_next = np.argmax(self.target_network(next_state_batch), axis=1)
                #actual_values = tf.where(done_batch, reward_batch, reward_batch + discount_factor*value_next)
                #real values s the target value that the network attempts to get closer to on each iteration.
            
                state_action_values = tf.math.reduce_sum(
                    self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions.n),
                    axis=1
                )
                
                loss_value = loss(actual_values, state_action_values)
            
            gradients = tape.gradient(loss_value, variables)
            optimizer.apply_gradients(zip(gradients, variables))

            if isinstance(loss_value, int):
                erro.append(loss_value)
            else:
                erro.append(loss_value.numpy())        
            
            i += 1
        return mean(erro)
    #DDQN DDQN DDQN DDQN DDQN
    def _apply_gradient_descent(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor, trace_length):
        
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=0.001)
        loss = tf.keras.losses.Huber()
        erro = list()
        
        #min_history = trace_length*1.5
        #if len(memory) < min_history:
        #    return 0
        i = 0
        while i < batch_size:
            #print("ANTES DE FORWARD", self.lstm_out_c, self.lstm_out_state)
            #trace_length -> tamanho da janela q vai ter cada ep do batch
            transitions = memory.sample_DRQN_pos_DODO_gambia(trace_length = trace_length )
            #print("NAO SEI MAIS", transitions)
            #print("NAO SEI MAIS MUDO", transitions)
            batch = DQNTransition(*zip(*transitions))
            #print("NAO SEI MAIS", batch)
            #variables = self.policy_network.trainable_variables

            #with tf.GradientTape() as tape:
            #    tape.watch(variables)

            state_batch = tf.convert_to_tensor(batch.state)
            #print(state_batch)
            action_batch = tf.convert_to_tensor(batch.action)
            reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
            next_state_batch = tf.convert_to_tensor(batch.next_state)
            #print(next_state_batch.shape)
            done_batch = tf.convert_to_tensor(batch.done, dtype=tf.float32)
            #print(done_batch)
            target_q = self.target_network(next_state_batch)
            main_q = self.policy_network(next_state_batch)
            #main_q = tf.stop_gradient(main_q)

            next_action = tf.argmax(main_q, axis=1)
            #target_value = tf.reduce_sum(tf.one_hot(next_action, self.n_actions.n) * target_q, axis=1)
            target_value = tf.reduce_sum(tf.multiply(target_q,tf.one_hot(next_action, self.n_actions.n)), axis=1)
            actual_values = (1-done_batch) *  discount_factor * target_value + reward_batch

                #value_next = np.argmax(self.target_network(next_state_batch), axis=1)
                #actual_values = tf.where(done_batch, reward_batch, reward_batch + discount_factor*value_next)
                #real values s the target value that the network attempts to get closer to on each iteration.
                
            with tf.GradientTape() as tape:
                #state_action_values = tf.math.reduce_sum(
                #    self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions.n),
                #    axis=1
                #)
                state_action_values = tf.reduce_sum(tf.multiply(
                    self.policy_network(state_batch),tf.one_hot(action_batch, self.n_actions.n)),
                    axis=1
                )
                loss_value = loss(actual_values, state_action_values)
            
            gradients = tape.gradient(loss_value, self.policy_network.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

            if isinstance(loss_value, int):
                erro.append(loss_value)
            else:
                erro.append(loss_value.numpy())        
            
            i += 1
        return mean(erro)

    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size',1) #episodios
        discount_factor: float =  0.99
        learning_rate: float = kwargs.get('learning_rate', 0.0001)        
        update_target_every: int = kwargs.get('update_target_every',1440) #700 8000
        memory_capacity: int = kwargs.get('memory_capacity', 14400) #4514000 para 1 ano 
        
        #self.window_steps_per_episode = 4
        trace_length = self.window_steps_per_episode #OLHA NO INIT DA CLASS        

        current_time = datetime.now().strftime("%Y%m%d - %H%M%S")
        log_dir = 'logs/drqn/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        #https://observablehq.com/@katnoria/visualising-epsilon-decay
        epsilon = 1.0
        #decay = 0.996 #750
        #decay = 0.9974 #1000
        #decay = 0.991 #teste 19/05/2021
        #decay = 0.99938 #5000
        #decay = 0.9984 #1500
        #decay = 0.991
        decay = 0.998 #1500 26/05/2021
        #decay = 0.99745 #1500 25/06/2021 random ate 900
        min_epsilon = 0.1

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        
        episode = 0
        total_steps_done = 0
        total_reward_env = 0
        stop_training = False
        total_rewards = np.empty(n_episodes)
        

        print('====      AGENT ID: {}      ===='.format(self.id))

        #while episode < n_episodes and not stop_training:
        for n in range(n_episodes):
            episode = n
            state = self.env.reset()
            done = False
            steps_done = 0
            rewards_episode = 0
            losses = list()
            #self.lstm.reset_states(states=[np.zeros((4, 256)), np.zeros((4, 256))])
            #epsilon = 0.99
            epsilon = max(min_epsilon, epsilon * decay)
            
            #//self.policy_network.get_layer(name = 'lstm').reset_states(states=None)
            
            print(self.policy_network.get_layer(name = 'lstm'))
            	    
            #for valor in range(self.window_size_from_env-1):
            #    acao = 0
            #    prox_estado, recomp, doine, _ = self.env.step(acao)
            #    state = prox_estado
            #print("primeiro dado da rede", state)
            while not done:
                
                action = self.get_action(state, threshold=epsilon)
                next_state, reward, done, info = self.env.step(action)
                #print(info['portfolio'][0])
                #print("AQUI SAI DO ENV", state)
                memory.push(state, action, reward, next_state, done)

                state_done = state
                state = next_state
                total_reward_env += reward
                rewards_episode += reward
                steps_done += 1
                total_steps_done +=1
                money = info['portfolio'][0]
                #print(steps_done)
                if len(memory) > self.window_steps_per_episode:                   
                    loss = self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor,trace_length)
                    #print(loss)
                    losses.append(loss)   
                    #if isinstance(loss, int):
                    #    losses.append(loss)
                    #else:
                    #    losses.append(loss.numpy())
                    
                    if steps_done % update_target_every == 0:
                        print("MEMORY LEN", len(memory))
                        print("UPDATING TARGET NETWORK")
                        self.target_network.set_weights(self.policy_network.get_weights())
                        #self.target_network = tf.keras.models.clone_model(self.policy_network)
                        #self.target_network.trainable = False             

                        #performance = pd.DataFrame.from_dict(self.env.action_scheme.portfolio.performance, orient='index')

                   
                    if n_steps and steps_done >= n_steps:
                        done = True
                        memory.pop()
                        memory.push(state_done, action, reward, next_state, done)
                        #print("AQUI é topo pilha", state_done, action, reward, next_state, done)
                        #//self.policy_network.get_layer(name = 'lstm').reset_states(states=None)
                        self.env.reset()
                
                
            
            #money = performance.net_worth

            mean_losses = mean(losses)
            total_rewards[n] = rewards_episode            
            avg_rewards = total_rewards.mean()
            with summary_writer.as_default():
                tf.summary.scalar('episode reward', rewards_episode, step=n)
                tf.summary.scalar('running avg reward', avg_rewards, step=n)
                tf.summary.scalar('loss', mean_losses, step=n)
                tf.summary.scalar('Net_Worth',money, step=n)
                tf.summary.scalar('last_loss_batch', losses[-1], step=n)
                tf.summary.scalar('Epsilon', epsilon, step=n)
            if n % 1 == 0:
                print("episode:", n, "episode reward:", rewards_episode,  "avg reward:", avg_rewards,
                  "episode loss: ", mean_losses,  'last_loss_batch:', losses[-1], 'Net_Worth:',money)    
            #if n % 1 == 0:
            #    print("episode:", n, "episode reward:", rewards_episode)
        

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                filename = self.save(save_path, episode=episode)

            
        

        return filename

    def test(self,
              n_steps: int = None,
              n_episodes: int = None                            
              ):
        #https://observablehq.com/@katnoria/visualising-epsilon-decay
        epsilon = 0.001        
        #memory_capacity =  8000
        #memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        current_time = datetime.now().strftime("%Y%m%d - %H%M%S")
        log_dir = 'logs/EVAL/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)
        
        episode = 0
        total_steps_done = 0
        total_reward_env = 0
        stop_training = False
        total_rewards = np.empty(n_episodes)
        
        print('====      AGENT ID: {}      ===='.format(self.id))

        #while episode < n_episodes and not stop_training:
        for n in range(n_episodes):
            episode = n
            state = self.env.reset()
            done = False
            steps_done = 0
            rewards_episode = 0
            losses = list()
            #self.lstm.reset_states(states=[np.zeros((4, 256)), np.zeros((4, 256))])
            #epsilon = 0.99
            #epsilon = max(min_epsilon, epsilon * decay)
            
            #//self.policy_network.get_layer(name = 'lstm').reset_states(states=None)
            
            #print(self.policy_network.get_layer(name = 'lstm'))
	    
            #for valor in range(self.window_size_from_env-1):
            #    acao = 0
            #    prox_estado, recomp, doine, _ = self.env.step(acao)
            #    state = prox_estado
            #print("primeiro dado da rede", state)
            while not done:
                
                action = self.get_action(state, threshold=epsilon)
                next_state, reward, done, info = self.env.step(action)
                #print(info['portfolio'][0])
                #print("AQUI SAI DO ENV", state, action, reward, next_state, done)
                #memory.push(state, action, reward, next_state, done)

                state_done = state
                state = next_state
                total_reward_env += reward
                rewards_episode += reward
                steps_done += 1
                total_steps_done +=1
                money = info['portfolio'][0]

                #if len(memory) > self.window_steps_per_episode:                   
                #    loss = self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor,trace_length)
                #    losses.append(loss)   
                    #if isinstance(loss, int):
                    #    losses.append(loss)
                    #else:
                    #    losses.append(loss.numpy())
                    
                #    if steps_done % update_target_every == 0:
                #        print("MEMORY LEN", len(memory))
                #        print("UPDATING TARGET NETWORK")
                #        self.target_network.set_weights(self.policy_network.get_weights())
                        #self.target_network = tf.keras.models.clone_model(self.policy_network)
                        #self.target_network.trainable = False             

                        #performance = pd.DataFrame.from_dict(self.env.action_scheme.portfolio.performance, orient='index')

                    # if render_interval is not None and steps_done % render_interval == 0:
                    #     self.env.render(
                    #         episode=episode,
                    #         max_episodes=n_episodes,
                    #         max_steps=n_steps
                    #     )

                if n_steps and steps_done >= n_steps:
                    done = True
                #        memory.pop()
                #        memory.push(state_done, action, reward, next_state, done)
                        #print("AQUI é topo pilha", state_done, action, reward, next_state, done)
                        #//self.policy_network.get_layer(name = 'lstm').reset_states(states=None)
                    self.env.reset()               
                
            
            #money = performance.net_worth

            #mean_losses = mean(losses)
            total_rewards[n] = rewards_episode            
            avg_rewards = total_rewards.mean()
            with summary_writer.as_default():
                tf.summary.scalar('episode reward', rewards_episode, step=n)
                tf.summary.scalar('running avg reward', avg_rewards, step=n)
            #    tf.summary.scalar('loss', mean_losses, step=n)
                tf.summary.scalar('Net_Worth',money, step=n)
            #    tf.summary.scalar('last_loss_batch', losses[-1], step=n)
            #    tf.summary.scalar('Epsilon', epsilon, step=n)
            if n % 1 == 0:
                print("episode:", n, "episode reward:", rewards_episode,  "avg reward:", avg_rewards,
                     'Net_Worth:',money)    
            #if n % 1 == 0:
            #    print("episode:", n, "episode reward:", rewards_episode)
        

            #is_checkpoint = save_every and episode % save_every == 0

            #if save_path and (is_checkpoint or episode == n_episodes - 1):
            #    self.save(save_path, episode=episode)            

        mean_reward = total_reward_env / steps_done

        return mean_reward
