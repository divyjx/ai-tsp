import numpy as np
import sys 
import time
import random
import math
import copy


f=open(sys.argv[1], "r")
start = time.time() 
# f=open("noneuc_250","r")
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

#initializations
distances = np.array(list(map(conv,disMat)))
initialPheromone=10/number
pheromones = [[(initialPheromone )for i in range(number)] for j in range(number) ]
# for x in range(number):
#     for y in range(number):
#         if x!=y:
#             pheromones[x][y]+=1/distances[x][y]
numberOfAnts = number//10
# if number==100 and name=="euclidean":
#     alpha = 3
#     beta = 3
#     rho = 0.1
#     Q = 0.1
# else:
#     alpha = 40
#     beta = 40
#     rho = 0.1
#     Q = 0.1
alpha = 20
beta = 20
# rho = 0.001*number
rho = 0.1
# rho = (0.001*number)%1
Q = 1
e=0
if name == "euclidean":
    e=1
antTime=(number//2 + 50)*e + 0*(not e)*280
antTime=(number//2 + 50)
antTime=(200)
print(antTime)
bestCost = float('Inf')
bestPath = []

def tourCost(path):
    cost = 0
    for i in range(number):
        cost += distances[path[i]][path[(i+1)%number]]
    return cost


# Ant Colony Optimization
try:
    while (time.time() - start) < antTime:
        allAntsPath = []
        delta_pheromones = [[0 for x in range(number)] for y in range(number)]
        for k in range(numberOfAnts):
            unvisitedCities = list(range(0,number))
            startCity = random.randint(0,number-1)
            unvisitedCities.remove(startCity)
            antPath = [startCity]
            # tour of an ant
            while len(antPath) < number:
                i = antPath[-1]
                # amount of pheromone and visibility determine probability
                ph_ij = [(pheromones[i][j]**alpha * ((1/distances[i][j])**beta)) for j in unvisitedCities]
                probList_ij = [x/sum(ph_ij) for x in ph_ij]
                j = random.choices(unvisitedCities, weights=probList_ij)[0]
                antPath.append(j)
                unvisitedCities.remove(j)

            allAntsPath.append(antPath)
            c = tourCost(antPath)
            if c < bestCost:
                bestCost = c
                bestPath = antPath
                print(f"Best Cost: {c} ")
                # print(bestPath)
        
        allAntsPath.sort(key=lambda ap : tourCost(ap))    
        # top 20% tours
        # top = int(0.2*len(allAntsPath)+1)
        top = int(0.2*len(allAntsPath)+1)
        # for path in allAntsPath[:top]:
        for path in allAntsPath:
            for i in range(number):
                j = path[(i+1)%number]
                delta_pheromones[path[i]][j] += Q / distances[path[i]][j]
        
        #updating pheromone concentration
        for i in range(number):
            for j in range(number):
                # pheromone at t=t+1
                pheromones[i][j] = (1-rho)*pheromones[i][j] + delta_pheromones[i][j]


except KeyboardInterrupt as e:
    print("Interrupted on user demand.")
    print(bestCost)
    print(bestPath)

# print("\n",pheromones)
try:
        print("Starting 2-Opt and 3-Opt")
        Visited = np.array(bestPath)
        # start = time.time()
        end = time.time()
        curr = Visited
        neighbours=0
        while (time.time() - start < 280):
            p = random.random()
            p = 0.7
            del neighbours
            neighbours = copy.deepcopy(curr)
            x = np.random.randint(0, number-1)
            y = np.random.randint(0, number-1)
            z = np.random.randint(0, number-1)
            m = min(x, y, z)
            M = max(x, y, z)
            num=sorted([x,y,z])
            if p <= 0.4:  # 2 swap
                temp = neighbours[x]
                neighbours[x] = neighbours[y]
                neighbours[y] = temp
            elif p <= 0.6:  # 3 swap
                temp = neighbours[x]
                neighbours[x] = neighbours[y]
                neighbours[y] = neighbours[z]
                neighbours[z] = temp
            elif p <= 0.8:  # 3 swap
                neighbours[m:M]= neighbours[m:M][::-1]
            else:  # insert
                temp = neighbours[m]
                for i in range(m, M):
                    neighbours[i] = neighbours[i+1]
                neighbours[M] = temp
            if (tourCost(neighbours) -  tourCost(curr)) < 0:
                curr = neighbours
                print(tourCost(curr))
            end = time.time()
except KeyboardInterrupt:
    print(tourCost(curr), end-start)
# print(*curr)
print ("Best Cost : " ,tourCost(curr))
print (list(curr))
