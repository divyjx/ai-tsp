import numpy as np
import sys
import time
import random
import copy
start = time.time()


f = open(sys.argv[1], "r")
Cities = []
name = f.readline().rstrip("\n")
number = int(f.readline().rstrip("\n"))

for i in range(number):
    x = f.readline().rstrip("\n").split()
    Cities.append(x)


def conv(lis):
    new = []
    for x in lis:
        new.append(float(x))
    return new


nC = list(map(conv, Cities))

np.set_printoptions(precision=11)
nC = np.array(nC)
global distances
disMat = []
for i in range(number):
    x = f.readline().rstrip("\n").split()
    disMat.append(x)

# initializations
distances = np.array(list(map(conv, disMat)))
pheromones = [[0.5 for i in range(number)] for j in range(number)]
numberOfAnts = 10
endTime = 200
if number == 100 and name == "euclidean":
    numberOfAnts = 100
    alpha = 3
    beta = 3
    rho = 0.1
    Q = 0.1

elif number==250 and name=="euclidean":
    endTime = 60
    numberOfAnts = 20
    alpha = 50
    beta = 50
    rho = 0.1
    Q = 0.1
elif number==500 and name=="euclidean":
    endTime = 60
    numberOfAnts = 10
    alpha = 20
    beta = 20
    rho = 0.1
    Q = 0.1
elif number==100 and name=="noneuclidean":
    numberOfAnts = 100
    alpha = 40
    beta = 40
    rho = 0.1
    Q = 0.1
elif number == 500 and name == "noneuclidean":
    numberOfAnts = 10
    alpha = 80
    beta = 80
    rho = 0.1
    Q = 0.1
else:
    numberOfAnts=75
    alpha=40
    beta = 40
    rho=0.1
    Q=0.1


bestCost = float('Inf')
bestPath = []


def tourCost(path):
    cost = 0
    for i in range(number):
        cost += distances[path[i]][path[(i+1) % number]]
    return cost


# Ant Colony Optimization
try:
    while time.time() - start < endTime:
        allAntsPath = []
        delta_pheromones = [[0 for x in range(number)] for y in range(number)]
        for k in range(numberOfAnts):
            unvisitedCities = list(range(0, number))
            startCity = random.randint(0, number-1)
            unvisitedCities.remove(startCity)
            antPath = [startCity]
            # tour of an ant
            while len(antPath) < number:
                i = antPath[-1]
                # amount of pheromone and visibility determine probability
                ph_ij = [(pheromones[i][j]**alpha * ((1/distances[i][j])**beta))
                         for j in unvisitedCities]
                probList_ij = [x/sum(ph_ij) for x in ph_ij]
                j = random.choices(unvisitedCities, weights=probList_ij)[0]
                antPath.append(j)
                unvisitedCities.remove(j)

            allAntsPath.append(antPath)
            c = tourCost(antPath)
            if c < bestCost:
                bestCost = c
                bestPath = antPath
                print(f"Best Cost: {c} *** Time: {time.time()-start}\n")
                print(bestPath, end='\n\n')

        allAntsPath.sort(key=lambda ap: tourCost(ap))
        # top 20% tours
        top = int(0.2*len(allAntsPath)+1)
        for path in allAntsPath[:top]:
            for i in range(number):
                j = path[(i+1) % number]
                delta_pheromones[path[i]][j] += Q / distances[path[i]][j]

        # updating pheromone concentration
        for i in range(number):
            for j in range(number):
                # pheromone at t=t+1
                pheromones[i][j] = (1-rho)*pheromones[i][j] + \
                    delta_pheromones[i][j]

except ValueError as v:
    print(bestCost)
    print(bestPath)
except KeyboardInterrupt as e:
    print("Interrupted on user demand.")
    print(bestCost)
    print(bestPath)

try:
    print("Starting 2-Opt and 3-Opt")
    curr = bestPath
    currCost = bestCost
    neighbours = 0
    while (time.time() - start < 300):
        p = random.random()
        del neighbours
        neighbours = copy.deepcopy(curr)
        while (True):
            r = sorted(random.choices([i for i in range(number-1)], k=3))
            x, y, z = r
            if (y-x) >= 2 and (z-y) >= 2 and (x-z+number) % number >= 2:
                break

        m = min(r)
        M = max(r)

        if p < 0.2:  # 2 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = temp
        elif p < 0.4:  # 3 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = neighbours[z]
            neighbours[z] = temp
        elif p < 0.6:  # 2-opt
            neighbours[m:M] = neighbours[m:M][::-1]
        elif p < 0.8:  # 3-opt
            c1 = neighbours[x+1:y+1]
            c2 = neighbours[y+1:z+1]
            c3 = neighbours[z+1:] + neighbours[:x+1]
            n = []
            n.append(c1+c2[::-1]+c3[::-1])     # c1c2'c3'
            n.append(c1+c3+c2[::-1])           # c1c3c2'
            n.append(c1+c3+c2)                 # c1c3c2
            n.append(c1+c3[::-1]+c2[::-1])     # c1c3'c2'
            n.append(c1+c3[::-1]+c2)           # c1c3'c2
            n.append(c1+c3+c2[::-1])           # c1c3c2'

            for x in n:
                if tourCost(x) < tourCost(neighbours):
                    neighbours = x

        else:  # insert
            temp = neighbours[m]
            for i in range(m, M):
                neighbours[i] = neighbours[i+1]
            neighbours[M] = temp
        if (tourCost(neighbours) - currCost) < 0:
            currCost = tourCost(neighbours)
            curr = neighbours
            print("Best Cost : ", currCost)
            print(neighbours, end="\n\n")
except KeyboardInterrupt:
    pass
print("Final Best Cost : ", currCost)
print(curr)
