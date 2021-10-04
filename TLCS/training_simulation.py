import traci
import numpy as np
import random
import timeit
import os
import torch

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

# Duration_NS = 12
# Duration_NSL = 9
# Duration_EW = 16
# Duration_EWL = 7

# Duration_NS = 18
# Duration_NSL = 7
# Duration_EW = 16
# Duration_EWL = 7

# Duration_NS = 30
# Duration_NSL = 11
# Duration_EW = 22
# Duration_EWL = 12

Duration_NS = 47
Duration_NSL = 23
Duration_EW = 36
Duration_EWL = 17

O = [Duration_NS,Duration_NSL,Duration_EW,Duration_EWL]


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs,batch_size):
        self._model = Model
        self._Model_A = Model.critic
        self._Model_B = Model.critic_target
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps 
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_store = []
        self._speed_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._loss_store = []
        self._training_epochs = training_epochs
        self._batch_size = batch_size
        self.history = 0
        self.loss_value = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a = 0

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._sum_speed = 0
        old_total_wait = 0
        old_action = 0
        reward = 0
        re = 0
        current_phase = 0
        self.reward = 0
        duration = [Duration_NS,Duration_NSL,Duration_EW,Duration_EWL]
        old_duration = [Duration_NS,Duration_NSL,Duration_EW,Duration_EWL]
        self.a = 0
        
        # self._simulate(100)
        old_state = self._get_state()

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()    

            reward = - current_total_wait
            # reward = old_total_wait - current_total_wait
            # print("reward:",reward)
            
    
            current_phase = int(traci.trafficlight.getPhase("TL")/2)
            phase = current_phase - 1
            if phase == -1:
                phase = 3
                
            action = self._choose_action_A(current_state, epsilon,current_phase,duration)# phase, epsilon)

            
            # print(phase, old_action, old_duration, duration)
            # saving the data into the memory
            if self._step != 0:
                x = [0,0,0,0]
                y = [0,0,0,0]
                x[phase] = 1
                y[phase] = 1         
                if self._step < self._max_steps - self._green_duration:
                    self._Memory.add_sample((old_state, old_action, reward, current_state,0,x,y,old_duration,duration))
                else:
                    self._Memory.add_sample((old_state, old_action, reward, current_state,1,x,y,old_duration,duration))

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:# and i == 0:
                self._set_yellow_phase(current_phase)
                self._simulate(self._yellow_duration)
                
            for i in range(4): 
                old_duration[i] = duration[i]
                
            current_phase += 1 
            if current_phase == 4:
                current_phase = 0
     
            green = duration[current_phase]
            a = 6*action - 6 + green #- 6*i
            # if a > O[current_phase] + 2*3:
            #     a = O[current_phase] + 2*3
            # if a < O[current_phase] - 2*3:
            #     a = O[current_phase] - 2*3
            if a < 7: 
                a = 7   # min green
            # duration[current_phase] = a

            self.a = current_phase
                
            # execute the phase selected before
            ###!!!!###
            # print(current_phase,a)
            self._set_green_phase(current_phase)
            self._simulate(a)
            # print("current phase:",current_phase,"green:",a)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            # if reward < 0:
            self._sum_neg_reward += reward
            re += 1
            self.reward = self._sum_neg_reward/re
                
        print("total queue:",self._sum_queue_length, "  ", "Total Reward:", self.reward, " ", "average Speed:",self._sum_speed/self._max_steps)

        self._save_episode_stats()
        print("Total negative reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        self.loss_value = 0
        
        self._Model_A.to(self.device)
        self._Model_B.to(self.device)
        
        for _ in range(self._training_epochs):
            self._replay()
            self.loss_value += self.history
            if ( _ / 50 == 0):
                self._model._update_model()
        self.loss_value = torch.tensor(self.loss_value).to("cpu")
        self._save_loss()
        print("loss:",self.loss_value)
        training_time = round(timeit.default_timer() - start_time, 1)
        
        self._Model_A.to("cpu")
        self._Model_B.to("cpu")

        return simulation_time, training_time


    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            speed = self._get_speed()
            self._sum_speed += speed
        
    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        self._waiting_times = {}
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time 
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
                
        if len(self._waiting_times) == 0: 
            total_waiting_time = 0
        else: 
            total_waiting_time = sum(self._waiting_times.values())/len(self._waiting_times)
        return total_waiting_time 


    def _choose_action_A(self, state, epsilon,phase,old_duration): #phase, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            x = [0,0,0,0]
            x[phase] = 1
            return torch.argmax(self._model.predict(state,x,old_duration))#,phase)) # the best action given the current state
    
    '''
    def _choose_action_B(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model_B.predict_one(state)) # the best action given the current state
    '''

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def get_green(self,current_phase):
        if current_phase == 0:
            green = Duration_NS
        elif current_phase == 1:
            green = Duration_NSL
        elif current_phase == 2:
            green = Duration_EW
        else: 
            green = Duration_EWL
        return green

    def _get_queue_length(self): 
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length
    
    def _get_speed(self):
        total_speed = 0
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_speed = traci.vehicle.getSpeed(car_id)
            total_speed +=car_speed
        if len(car_list) == 0: 
            s = 0
        else: 
            s = total_speed/len(car_list)
        return s
            
    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros((3,14,100))
        lane = ["N2TL_0","N2TL_1","N2TL_2","E2TL_0","E2TL_1","E2TL_2","E2TL_3","S2TL_0","S2TL_1","S2TL_2","W2TL_0","W2TL_1","W2TL_2","W2TL_3"]
        car_list = traci.vehicle.getIDList()
        lane_group = 0

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            car_speed = traci.vehicle.getSpeed(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            lane_cell = int(lane_pos/7.5)
            
            for i in range(len(lane)):
                if lane_id == lane[i]:
                    lane_group = i

            # _ = int(str(lane_group)+str(lane_cell))

            # if car_speed == 0:
            #     state[_] = - (traci.vehicle.getAccumulatedWaitingTime(car_id) + 1) 
            # else:
            #     state[_] = car_speed 
            
            # if car_speed == 0:
            #     state[lane_group][lane_cell] = - (traci.vehicle.getAccumulatedWaitingTime(car_id) + 1) 
            # else:
            #     state[lane_group][lane_cell] = car_speed 
            
            state[0][lane_group][lane_cell] = 1
            state[1][lane_group][lane_cell] = car_speed
            state[2][lane_group][lane_cell] = traci.vehicle.getAccumulatedWaitingTime(car_id)
        
      
        # state = np.zeros((29))
        # lane = ["N2TL_0","N2TL_1","N2TL_2","E2TL_0","E2TL_1","E2TL_2","E2TL_3","S2TL_0","S2TL_1","S2TL_2","W2TL_0","W2TL_1","W2TL_2","W2TL_3"]
        # car_list = traci.vehicle.getIDList()
        # lane_group = 0

        # for car_id in car_list:
        #     lane_pos = traci.vehicle.getLanePosition(car_id)
        #     car_speed = traci.vehicle.getSpeed(car_id)
        #     lane_id = traci.vehicle.getLaneID(car_id)
        #     lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

        #     # distance in meters from the traffic light -> mapping into cells
        #     lane_cell = int(lane_pos/7.5)
            
        #     for i in range(len(lane)):
        #         if lane_id == lane[i]:
        #             lane_group = i
            
        #     # state[0][lane_group][lane_cell] = 1
        #     # state[1][lane_group][lane_cell] = car_speed
        #     # state[2][lane_group][lane_cell] = traci.vehicle.getAccumulatedWaitingTime(car_id)
            
        #     if lane_cell <= 10: 
        #         state[lane_group] += 1
                
        #     state[lane_group+14] += 1
        
        # state[28] = self.a
        # # print("State:",state)
        return state
   
    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        Model = self._model
        batch = self._Memory.get_samples(self._batch_size)   
        batch = np.array(batch)
 
        if len(batch) > 0:  # if the memory is full enough
            # states = np.array([val[0] for val in batch])  # extract states from the batch
            # next_states = np.array([val[3] for val in batch])  # extract next states from the batch
            # rewards = np.array([val[2] for val in batch]).reshape(100,1)
            states = torch.tensor(([val[0] for val in batch]),dtype=torch.float).to(self.device)  # extract states from the batch
            actions = torch.tensor(([val[1] for val in batch]),dtype=torch.float).to(self.device)
            rewards = torch.tensor(([val[2] for val in batch]),dtype=torch.float).reshape(-1,1).to(self.device)
            next_states = torch.tensor(([val[3] for val in batch]),dtype=torch.float).to(self.device)  # extract next states from the batch
            dones = torch.tensor(([val[4] for val in batch]),dtype=torch.float).reshape(-1,1).to(self.device)
            a = torch.tensor(([val[5] for val in batch]),dtype=torch.float).to(self.device)
            b = torch.tensor(([val[6] for val in batch]),dtype=torch.float).to(self.device)
            c = torch.tensor(([val[7] for val in batch]),dtype=torch.float).to(self.device)
            d = torch.tensor(([val[8] for val in batch]),dtype=torch.float).to(self.device)

            self.history = Model.critic_learn(states,actions,rewards,next_states,self._gamma,dones,a,b,c,d).squeeze(0).detach()  # train the NN  


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self.reward)  # how much negative reward in this episode
        self._speed_store.append(self._sum_speed / self._max_steps)
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode
        
    def _save_loss(self):
        self._loss_store.append(self.loss_value)

    @property
    def loss_store(self):
        return self._loss_store
    
    @property
    def reward_store(self):
        return self._reward_store

    @property
    def speed_store(self):
        return self._speed_store
    
    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store

