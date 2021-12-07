import numpy as np
import time
from tkinter import *
from functools import partial


# Representation on agent and robot map
# 1 : robot, 1 : object A  A, -1 : object B, 0 : no object
OBJA = 'A'
OBJB = 'B'
OBJC = 'C'
VIDE = ' '
ROBOT = 'R'

MEMORY_SIZE = 10 # Memory size of the agent 
SPREAD_DISTANCE = 5 # Distance within which the pheromon will spred with decreasing intensity
EVAPORATION_RATE = 0.25 # Rate at which the pheromon will evaporate into
WAIT_TIME = 0 # Time during which a robot will wait before giving up the call signal
IGNORE_TIME = 5 # Time during which the agent will ignore pheromon signals 

NA = 30 # Number of A items 
NB = 30 # Number of A items 
NC = 30 # Number of C items 
NR = 10  # Number of Robots 
HEIGHT = 15 # Map Height
LENGHT = 20 # Map Lenght
PAS = 1 # distance moovement from a robot 

IT = 2_000_000

# Take or Put constant variables used in put_ob and put obj functions
KT = 0.1
KP = 0.3
ER = 0 # Error rate 

# List and dic of agents 
LISTE_AGENT = []
DIC_AGENT = {}


# Env class
class Environnement():
    def __init__(self, height, lenght, nA, nB, nC, nR):
        # 2 grids are used one for object and one to avoid robot collision 
        self.object_grid = self.init_object_grid(height, lenght, nA, nB, nC)
        self.pheromon_grid = np.zeros((height,lenght))
        self.dic_agent_pos = self.init_agent_pos(height, lenght, nR)
        self.height = height
        self.lenght = lenght

    def init_object_grid(self, height, lenght, nA, nB, nC):
        grid = np.full((height, lenght),' ')
        ind = np.random.choice(height*lenght, size=nA+nB+nC, replace=False)

        # Modify the array returned by ravel which is a view of the original array, place nA objA and nB objB
        grid.ravel()[ind[:nA]] = np.array([OBJA for i in range(nA)])
        grid.ravel()[ind[nA:nA+nB]] = np.array([OBJB for j in range(nB)])
        grid.ravel()[ind[nA+nB:]] = np.array([OBJC for j in range(nC)])

        return grid

    def init_agent_pos(self, height, lenght, nR):
        # Create a dictionary with initial pos of robot 
        ind = np.random.choice(height*lenght, size=nR, replace=False)
        agent_pos = {i+1: (ind[i]%lenght,ind[i]//lenght) for i in range(nR)}
        return agent_pos

    def get_object(self, pos):
        x,y = pos
        return self.object_grid[y][x]

    def get_pos_agents(self,pos):
        x,y = pos
        agents_id = []
        for id in self.dic_agent_pos.keys():
            if self.dic_agent_pos[id] == (x,y) :
                agents_id.append(id)
        return agents_id 

    def get_cells_in_range(self,pos,reach):
        x,y = pos
        possible_neighbour = [(i,j) for i in range(-reach,reach+1,1) for j in range(-reach,reach+1,1)] # 8 directions, all cells in range range, unique only 
        cells_in_range = []
        for i,j in possible_neighbour :
            if (x+i) >= 0 and (x+i) < self.lenght and (y+j) >= 0 and (y+j) < self.height : # If (x+i,y+j) in bound and a 'valid' cell
                cells_in_range.append((x+i,y+j))
        return cells_in_range

    def get_pheromon(self,pos):
        x,y = pos
        return self.pheromon_grid[y][x]

    def evaporate(self):
        self.pheromon_grid = self.pheromon_grid *(1-EVAPORATION_RATE)

    def put_obj(self, obj, pos):
        x,y = pos
        self.object_grid[y][x] = obj

    def take_obj(self, pos):
        x,y = pos
        self.object_grid[y][x] = VIDE

    def update_agent_pos(self,id,pos):
        self.dic_agent_pos[id] = pos
        
    def eval_sorting(self):
        score = {OBJA:0, OBJB:0, OBJC:0}
        for i in range(self.height):
            for j in range(self.lenght):
                obj = self.object_grid[i][j]
                if obj != VIDE : # if there is an object on there
                    if i == 0 and j == 0 : # Top left corner
                        neighb = [(0, 1), (1, 1), (1, 0)]
                    elif i == 0 and j == self.lenght-1 : # Top right corner
                        neighb = [(-1, 0), (0, 1), (-1, 1)]
                    elif i == self.height-1 and j == self.lenght-1 : # bottom right corner
                        neighb = [(-1, 0), (-1, -1), (0, -1)]
                    elif i == self.height-1 and j == 0 : # bottom left corner
                        neighb = [(1, 0), (1, -1), (0, -1)]
                    elif i == 0 : #  top side but not corner
                        neighb = [(-1, 0), (0, 1), (1, 1), (1, 0), (-1, 1)]
                    elif i == self.height-1 : # bot side but not corner
                        neighb = [(-1, 0), (-1, -1), (1, 0), (1, -1), (0, -1)]
                    elif j == self.lenght-1 : # right side but not corner
                        neighb = [(-1, 0), (-1, -1), (0, 1), (0, -1), (-1, 1)]
                    elif j == 0 : # left side but not corner
                         neighb = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
                    else :
                        neighb = [(-1, 0), (-1, -1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 1)]
                    value = 0
                    for k,l in neighb :
                        neighb_obj = self.object_grid[i+l][j+k]
                        if obj == neighb_obj :
                            value += 1
                        elif neighb_obj in [OBJA,OBJB,OBJC] : 
                            value -= 1
                        # CASE WHEN neighb_obj == VIDE does not directly reduce the score 
                    score[obj] += value/(len(neighb))   
        return (score[OBJA]/max(NA,1), score[OBJB]/max(NB,1), score[OBJC]/max(NC,1))

    # Display on the IPython Console    
    def representation(self):

        def get_agent(pos):
            agents_in_pos = []
            for agent in DIC_AGENT.values():
                if agent.pos == pos:
                    agents_in_pos.append(agent)
            return agents_in_pos 

        rep = ""
        for y in range(self.height):
            for x in range(self.lenght):
                pos = (x,y)
                if self.object_grid[y][x] == OBJA:
                    rep += "A "
                elif self.object_grid[y][x] == OBJB:
                    rep += "B "
                elif  self.object_grid[y][x] == OBJC:
                    rep += "C "
                elif pos in self.dic_agent_pos.values():
                    agents_in_pos = get_agent(pos)
                    sigle = "X "
                    for agent in agents_in_pos :
                        if agent.bag == OBJA:
                            sigle = "A "
                            break
                        elif agent.bag == OBJB:
                            sigle = "B "
                            break
                        elif agent.bag == OBJC:
                            sigle = "C " 
                            break
                    rep += sigle 
                else:
                    rep += "  "
            rep += ' \n'
        print(rep)
        print('\n \n')

# Agent class
class Agent():
    # The agent does only know its position, if there is an object on its feet and if it has an object on its bag 
    def __init__(self, env, id):
        self.id = id
        self.pos = ()
        self.memory = []
        self.bag = VIDE
        self.stand_by = False # Whether the robot is waiting for help to carry a C object 
        self.wait_time = 0 # Waiting time, after X sec, the robot will give up 
        self.following = False # If the robot is helping an other robot to carry a C object, then both will agree that the first robot will be leading 
        self.association = None # (idx,idy) ids of the robot in the association to carry a C object  
        self.give_up_time = 0
        self.spread_distance = SPREAD_DISTANCE
        self.env = env

    def init_pos(self):
        # Get the initial pos of the robot
        self.pos = self.env.dic_agent_pos[self.id]

    # Check if the moove is possible 
    def check_valid_dep(self,pos):
        x,y = pos 
        if x >= self.env.lenght or x < 0 or y >= self.env.height or y < 0 :
            return False
        return True

    def moove(self):
        # There is a chance that the agent does follow the pheromon gradient direction 
        x_grad,y_grad,pheromon = self.get_pheromon()
        if np.random.random() > (1-np.exp(-pheromon)) or self.bag == VIDE or self.give_up_time > 0 : # Probability to keep a random trajectory
            cells_in_range = self.env.get_cells_in_range(self.pos,1)
            cells_in_range.remove(self.pos)
            ind = np.random.randint(0,len(cells_in_range))
            pos = cells_in_range[ind]
        else : # follow the gradient 
            pos = (x_grad,y_grad)
        self.env.update_agent_pos(self.id,pos)
        self.pos = pos
        if self.association != None :
            associate_agent_id = self.association[1]
            self.env.update_agent_pos(associate_agent_id,pos)
            DIC_AGENT[associate_agent_id].pos = pos

    # if the bag of the agent is empty and the agent can take the object on his case with a certain probability    
    def take(self, obj, dist):
        prob = (KT/(KT+dist))**2

        if np.random.random() <= prob and self.bag == VIDE:
            if obj != OBJC :
                self.bag = obj
                self.env.take_obj(self.pos)
            # If the robot want to carry a C object 
            else :
                # Check if there is a robot willing to help
                help = self.check_help()
                if help :
                    self.bag = obj
                    self.env.take_obj(self.pos)
                else :
                    self.stand_by = True
                    self.call_help()
                return True
        return False
      
    # Check if there is a robot willing to help, on the pos
    def check_help(self):
        agents_id = self.env.get_pos_agents(self.pos)
        agents_id.remove(self.id)
        if len(agents_id) > 0 : # if there are other robots on the cell
            for id in agents_id :
                agent = DIC_AGENT[id]
                if agent.bag == VIDE and agent.following == False : # A robot can help 
                    self.stand_by = False
                    self.wait_time = 0
                    self.association = (self.id,id)
                    agent.association = (id,self.id)
                    agent.following = True
                    return True
        return False
    
    # Call other robots for help 
    def call_help(self):
        self.wait_time +=1
        cells_in_range = self.env.get_cells_in_range(self.pos,self.spread_distance)
        for cell in cells_in_range :
            cell_x,cell_y = cell[0],cell[1]
            dist = max(abs(self.pos[0]-cell_x),abs(self.pos[1]-cell_y))
            self.env.pheromon_grid[cell_y][cell_x] += 1/(dist+1)

    # Get pheromon in the 8 neighbour cells 
    def get_pheromon(self):
        neighbour_cells = self.env.get_cells_in_range(self.pos,1)
        pheromon_cells = [(cell[0],cell[1],self.env.get_pheromon(cell)) for cell in neighbour_cells if cell != self.pos ] # list of (x,y,pheromon) elt 
        return max(pheromon_cells,key=lambda x:x[2])

    # if the agent has an object and his case is empty, he can put his object on his case with a certain probability  
    def put(self, obj, dist):
        prob = (dist/(KP+dist))**2

        if np.random.random() <= prob and self.env.get_object(self.pos) == VIDE:
            self.bag = VIDE
            self.env.put_obj(obj, self.pos)
            if self.association != None :
                _,associate_id = self.association
                associate_agent = DIC_AGENT[associate_id]
                associate_agent.following = False
                associate_agent.association = None
                self.association = None
            return True
        return False

    # get the distribution of encountered objects
    def get_dist(self, obj):
        s = 0
        for elt in self.memory:
            if elt == obj:
                s += 1
            elif elt != VIDE:  # Introduction of mistakes
                s += ER

        dist = s/len(self.memory)
        return dist

    # The agent use the perception function that will trigger the selected action function
    def perception(self):
        if len(self.memory) == MEMORY_SIZE:
            self.memory.pop(0)
        on_pos_object = self.env.get_object(self.pos)
        self.memory.append(on_pos_object)

        self.give_up_time == max(self.give_up_time-1,0) # Decrease give_up_time by 1 

        if self.following == True :
            return "follow"

        if self.stand_by == True and self.wait_time < WAIT_TIME :
            self.call_help()
            return "call_help"

        if self.stand_by == True and self.wait_time == WAIT_TIME :
            self.stand_by = False
            self.wait_time = 0
            self.give_up_time = IGNORE_TIME
            self.moove()
            return "moove"

        if self.env.get_object(self.pos) == VIDE:
            if self.bag != VIDE:  # THEN len(memory) != 0:
                dist = self.get_dist(self.bag)
                put = self.put(self.bag, dist)
                if put:
                    return "put"
            self.moove()
            return "moove"
        else:
            if self.bag == VIDE:
                dist = self.get_dist(on_pos_object)
                take = self.take(on_pos_object, dist)
                if take:
                    return "take"
            self.moove()
            return "moove"


if __name__ == '__main__':

    start_time = time.time()
    np.random.seed(1) # Use random Seed for reproductibility does not work and I don't know why 
    Env = Environnement(HEIGHT, LENGHT, NA, NB, NC, NR)
    # We create the agent list
    for agent_id in range(1,NR+1):
        agent = Agent(Env,agent_id)
        agent.init_pos()
        LISTE_AGENT.append(agent)
        DIC_AGENT[agent_id] = agent

    for iteration in range(1,IT+1):
        np.random.shuffle(LISTE_AGENT) # At each iteration, the order in which the robot will act is randomized
        for n in range(NR):
            LISTE_AGENT[n].perception()
        Env.evaporate()
        if iteration in [10_000,20_000,50_000,80_000,100_000,200_000,300_000,400_000,500_000,1_000_000,2_000_000]:
            Env.representation()
            evaluation = Env.eval_sorting()
            print(f"Environnement object A : {NA}, object B :  {NB}, height : {HEIGHT}, lenght : {LENGHT}")
            print(f" Number of agents : {NR}, Number of iterations : {iteration}, k+ : {KT}, k- : {KP}, error rate : {ER}")
            print(f" ----- time taken : {time.time() - start_time} seconds -----")
            print(f"Evaluation of the sorting algorithm : {evaluation[0]} for object A, {evaluation[1]} for object B, {evaluation[2]} for object C")
