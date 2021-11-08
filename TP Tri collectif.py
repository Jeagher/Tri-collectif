import numpy as np
import time


# Representation on agent and robot map
# 1 : robot, 1 : object A  A, -1 : object B, 0 : no object
OBJA = 1 
OBJB = -1
VIDE = 0
ROBOT = 1

MEMORY_SIZE = 10 # memory size of the agent 

NA = 75 # Number of A items 
NB = 75 # Number of A items 
NR = 10  # Number of Robots 
HEIGHT = 30 # Map Height
LENGHT = 30 # Map Lenght
PAS = 1 # distance moovement from a robot 

IT = 2_000_000

# Take or Put constant variables used in put_ob and put obj functions
KT = 0.1
KP = 0.1
ER = 0 # Error rate 


# Env class
class Environnement():
    def __init__(self, height, lenght, nA, nB, nR):
        # 2 grids are used one for object and one to avoid robot collision 
        self.object_grid = self.init_object_grid(height, lenght, nA, nB)
        self.agent_grid = self.init_agent_grid(height, lenght, nR)
        self.height = height
        self.lenght = lenght

    def init_object_grid(self, height, lenght, nA, nB):

        grid = np.zeros((height, lenght), dtype=np.int8)
        ind = np.random.choice(height*lenght, size=nA+nB, replace=False)

        # Modify the array returned by ravel which is a view of the original array, place nA objA and nB objB
        grid.ravel()[ind[:nA]] = np.array([OBJA for i in range(nA)])
        grid.ravel()[ind[nA:]] = np.array([OBJB for j in range(nB)])

        return grid

    def init_agent_grid(self, height, lenght, nR):

        grid = np.zeros((height, lenght), dtype=np.int8)
        ind = np.random.choice(height*lenght, size=nR, replace=False)

        grid.ravel()[ind] = np.array([ROBOT for i in range(nR)])

        return grid

    def get_object(self, x, y):
        return self.object_grid[y][x]

    def get_agent(self, x, y):
        return self.agent_grid[y][x]

    def put_obj(self, obj, x, y):
        self.object_grid[y][x] = obj

    def take_obj(self, x, y):
        self.object_grid[y][x] = VIDE

    def update_agent_pos(self, x_prev, y_prev, x, y):
        self.agent_grid[y_prev][x_prev] = VIDE
        self.agent_grid[y][x] = ROBOT
        
    def eval_sorting(self):
        score = {-1:0,1:0}
        for i in range(self.height):
            for j in range(self.lenght):
                obj = self.object_grid[i][j]
                if obj != VIDE : # if there is an object on there
                    if obj == OBJA : # if there is an object A 
                        obj_type = 1
                    elif obj == OBJB :
                        obj_type = -1
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
                        value += self.object_grid[i+l][j+k] # ObjA = 1 and objB = -1
                    value = value*obj_type
                    score[obj_type] += value/(len(neighb))   
        return score[1]/NA, score[-1]/NB

    # Display on the IPython Console    
    def representation(self, liste_agent):

        def get_agent(liste_agent, x, y):
            for agent in liste_agent:
                if agent.pos == (x, y):
                    return agent
            return

        rep = ""
        for y in range(self.height):
            for x in range(self.lenght):
                if self.object_grid[y][x] == OBJA:
                    rep += "A "
                elif self.object_grid[y][x] == OBJB:
                    rep += "B "
                elif self.agent_grid[y][x] == ROBOT:
                    agent = get_agent(liste_agent, x, y)
                    if agent.bag == VIDE:
                        rep += "X "
                    elif agent.bag == OBJA:
                        rep += "A "
                    elif agent.bag == OBJB:
                        rep += "B "
                else:
                    rep += "  "
            rep += ' \n'

        print(rep)
        print('\n \n')
        # print(self.agent_grid)

# Agent class
class Agent():
    # The agent does only know its position, if there is an object on its feet and if it has an object on its bag 
    def __init__(self, env):
        self.pos = ()
        self.memory = []
        self.bag = VIDE
        self.env = env

    def init_pos(self, n):
        # get index of sqares where there should be a robot in the environnement
        index = (self.env.agent_grid == ROBOT).nonzero()
        x, y = index[1][n], index[0][n]
        self.pos = (x, y)

    # Check if the moove is possible 
    def check_valid_dep(self, x, y): 
        if x >= self.env.lenght or x < 0 or y >= self.env.height or y < 0 or self.env.get_agent(x, y) == ROBOT:
            return False
        return True

    def moove(self):
        moove_list = [(-1, 0), (-1, -1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, 1)]
        ind = np.random.randint(0,8)
        x_dep, y_dep = moove_list[ind][0], moove_list[ind][1] # random moovement
        x = self.pos[0] + x_dep
        y = self.pos[1] + y_dep

        if self.check_valid_dep(x, y):
            self.env.update_agent_pos(self.pos[0], self.pos[1], x, y)
            self.pos = (x, y)
            return True
        return False

    # if the bag of the agent is empty and the agent can take the object on his case with a certain probability    
    def take(self, obj, dist):
        prob = (KT/(KT+dist))**2

        if np.random.random() <= prob and self.bag == VIDE:
            self.bag = obj
            self.env.take_obj(self.pos[0], self.pos[1])
            return True
        return False

    # if the agent has an object and his case is empty, he can put his object on his case with a certain probability  
    def put(self, obj, dist):
        prob = (dist/(KP+dist))**2

        if np.random.random() <= prob and self.env.get_object(self.pos[0], self.pos[1]) == VIDE:
            self.bag = VIDE
            self.env.put_obj(obj, self.pos[0], self.pos[1])
            return True
        return False

    # get the distribution of encountered objects A and B 
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
        on_pos_object = self.env.get_object(self.pos[0], self.pos[1])
        self.memory.append(on_pos_object)

        if self.env.get_object(self.pos[0], self.pos[1]) == VIDE:
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
    Env = Environnement(HEIGHT, LENGHT, NA, NB, NR)
    # np.random.seed(1) # Use random Seed for reproductibility does not work i don't know why 
    liste_agent = []
    # We create the agent list
    for agent_id in range(NR):
        agent = Agent(Env)
        agent.init_pos(agent_id)
        liste_agent.append(agent)

    for iteration in range(1,IT+1):
        np.random.shuffle(liste_agent) # At each iteration, the order in which the robot will act is randomized
        for n in range(NR):
            liste_agent[n].perception()
        # time.sleep(0.01)
        if iteration in [10_000,20_000,50_000,80_000,100_000,200_000,300_000,400_000,500_000,1_000_000,2_000_000]:
            Env.representation(liste_agent)
            evaluation = Env.eval_sorting()
            print(f"Environnement object A : {NA}, object B :  {NB}, height : {HEIGHT}, lenght : {LENGHT}")
            print(f" Number of agents : {NR}, Number of iterations : {iteration}, k+ : {KT}, k- : {KP}, error rate : {ER}")
            print(f" ----- time taken : {time.time() - start_time} seconds -----")
            print(f"Evaluation of the sorting algorithm : {evaluation[0]} for object A, {evaluation[1]} for object B")
