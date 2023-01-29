import numpy as np
import sys 
import time
import random
import math


# Function for Ant Colony
class Problem():

    def __init__(self, distances, numAnts, alpha, beta, rho, Q):

# Initializations
        self.distances = distances
        self.numAnts = numAnts
        self.Q = Q
        self.beta = beta
        self.rho = rho
        self.alpha = alpha
        
        # self.KBest = int(0.1*number+1)

        self.pheromones = [[0.0001 for x in range(number)] for y in range(number)]
        
        self.bestCost = float('Inf')
        self.bestTour = range(number)

# Function for Optimization
    def AntColonyOptimization(self):

# Loop initialization
        try:
            while time.time()-start < 250:
                ants = []

                delta_pheromones = [[0 for x in range(number)] for y in range(number)]

                for j in range(self.numAnts):
                    ant = Ant(self.distances, self.pheromones, self.alpha, self.beta)
                    ants.append(ant)

                    if ant.pathCost(self.distances) < self.bestCost:
                        self.bestCost = ant.pathCost(self.distances)
                        self.bestTour = ant.currentPath
                        # self.lastChange = time.time()
                        print(f"Current Best Cost: {self.bestCost}")
                        # print(*self.bestTour, end="\n\n")
                        # print(*self.pheromones, end="\n\n")

                ants.sort(key=lambda city: city.pathCost(self.distances))
                
                # for ant in ants[:self.KBest]:
                for ant in ants:
                    for ind, currCity in enumerate(ant.currentPath):
                        nextCity = ant.currentPath[(ind+1)%number]
                        delta_pheromones[currCity][nextCity] += self.Q / distances[currCity][nextCity]

                for i in range(number):
                    for j in range(number):
                        # pheromone at t=t+n
                        self.pheromones[i][j] = (1-self.rho)*self.pheromones[i][j] + delta_pheromones[i][j]
                        

                # if time.time()-self.lastChange > 300:
                #     break
            else:
                print(self.bestCost)
                print(*self.bestTour)
        except KeyboardInterrupt as e:
            print ("Interrupted on user demand.")
            print(self.bestCost)
            print(*self.bestTour)

# Ant class structure defined

class Ant():

    def __init__(self, distances, pheromones, alpha, beta):
        self.currentPath = []
        self.getPath(distances, pheromones, alpha, beta)

# Get Path for ant
    def getPath(self, distances, pheromones, alpha, beta):

        startCity = random.randint(0, number-1)
        remainingCities = list(range(0, number))
        remainingCities.remove(startCity)

        self.currentPath.append(startCity)
        while(len(self.currentPath) < number):

            i = self.currentPath[-1]

            ph_ij = [(pheromones[i][j]*alpha * (1/distances[i][j])*beta) for j in remainingCities]

            probList_ij = [x/sum(ph_ij) for x in ph_ij]

            J = random.choices(remainingCities, weights=probList_ij)[0] # highest prob is 0th element 
            self.currentPath.append(J)
            remainingCities.remove(J)

# Path cost function
    def pathCost(self, distances):
        cost = 0
        for i in range(len(self.currentPath)):
            cost += distances[self.currentPath[i]][self.currentPath[(i+1) % number]]
        return cost


start = time.time()
f=open(sys.argv[1],"r")
# f=open("noneuc_500","r")
Cities=[]
name=f.readline().rstrip("\n")
number=int(f.readline().rstrip("\n"))

for i in range(number):
    x=f.readline().rstrip("\n").split()
    Cities.append(x)
# print(Cities)
def conv(lis):
    new=[]
    for x in lis:
        new.append(float(x))
    return new    
nC=list(map(conv,Cities))

np.set_printoptions(precision=11)
nC=np.array(nC)
global distances
# print(nC)
disMat=[]
for i in range(number):
    x=f.readline().rstrip("\n").split()
    disMat.append(x)

distances = np.array(list(map(conv,disMat)))

# problem = Problem(distances, number, 3, 3, 0.1, 0.1)
problem = Problem(distances, number, 1, 5, 0.6, 0.6)
e=open("eu_tree_edges.txt","r")
for i in range(number):
    x=list(map(int,e.readline().rstrip("\n").split()))
    problem.pheromones[x[0]][x[1]]+=1
    problem.pheromones[x[1]][x[0]]+=1
problem.AntColonyOptimization()