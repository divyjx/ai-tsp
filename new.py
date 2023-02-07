import numpy as np
import sys 
import time
import random
import copy
from collections import defaultdict

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
pheromones = [[0.1 for i in range(number)] for j in range(number)]

nDe = []
for x in range(len(distances)):
    for y in range(x+1, len(distances)):
        nDe.append([x, y, distances[x, y]])


def costEval(node):
    cost = 0
    for x in range(number):
        From = node[x]
        if x == (number-1):
            To = node[0]
        else:
            To = node[x+1]
        cost += distances[From][To]
    return cost


class Graph():
    def __init__(self, nV, edges) -> None:
        self.vertices = nV  # number
        self.edges = copy.deepcopy(edges)  # edges (a,b,w)

    rank = []

    def findParent(self, parent, i):
        if parent[i] != i:
            parent[i] = self.findParent(parent, parent[i])
        return parent[i]

    def union(self, parent, rank, x, y):
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1

    def Kruskal(self):
        result = []
        i = 0
        e = 0
        self.Kgraph = sorted(self.edges, key=lambda item: item[2])
        parent = [i for i in range(self.vertices)]
        rank = [0 for i in range(self.vertices)]

        while (e < (self.vertices-1)):
            u, v, w = self.Kgraph[i]
            i += 1
            x = self.findParent(parent, u)
            y = self.findParent(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        self.KruskalTreeEdges = result
        return result

    def oddFinder(self):
        self.eoCounter = [0 for _ in range(self.vertices)]
        for x in self.KruskalTreeEdges:
            self.eoCounter[x[0]] += 1
            self.eoCounter[x[1]] += 1
        self.oddVertices = []  # odd degree vertex names
        self.evenVertices = []  # even degree vertex names
        for x in range(len(self.eoCounter)):
            if self.eoCounter[x] % 2 == 0:
                self.evenVertices.append(x)
            else:
                self.oddVertices.append(x)
        self.OddTree = []
        for x in self.edges:
            if x[1] in self.oddVertices and x[0] in self.oddVertices:
                self.OddTree.append(x)
        return self.OddTree

    def perfectMatching(self):
        self.PMtree = sorted(self.OddTree, key=lambda item: item[2])
        self.eoCounter2 = copy.deepcopy(self.eoCounter)
        visited = []
        res = []
        for x in self.PMtree:
            if x[0] not in visited and x[1] not in visited:
                res.append(x)
                visited.append(x[0])
                visited.append(x[1])
        self.PMtree = copy.deepcopy(res)
        return (self.PMtree)

    def EulerianTreeFinder(self):
        self.EuTree = copy.deepcopy(self.KruskalTreeEdges)
        for x in self.PMtree:
            if x not in self.EuTree:
                self.EuTree.append(x)
        self.EuMatrix = np.zeros(shape=(self.vertices, self.vertices))
        for x in self.EuTree:
            self.EuMatrix[x[0]][x[1]] = x[2]
            self.EuMatrix[x[1]][x[0]] = x[2]
        # print(self.EuMatrix)
        cost = 0
        for x in self.EuTree:
            cost += x[2]

class Eulertour:
    def __init__(self, vertices, Edges, root) -> None :
        self.adj = defaultdict(list)
        self.vis = defaultdict(bool)
        self.vertices = vertices
        self.path = []
        self.edges = copy.deepcopy(Edges)
        self.addEdge()
        self.Euler = [0]*(2*vertices)
        self.root = root

    def addEdge(self):
        self.adj.clear
        self.vis.clear
        for u, v, w in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)

    def Tour(self, start, index):
        self.vis[start] = True
        self.Euler[index] = start
        index += 1
        for x in self.adj[start]:
            if not self.vis[x]:
                index = self.Tour(x, index)
                self.Euler[index] = x
                index += 1
        return index

    def getTour(self):
        self.addEdge()
        path = []
        index = 0
        self.Tour(self.root, index)
        for i in range(2*self.vertices):
            path.append(int(self.Euler[i]))
        visi = []
        self.path=path
        for x in path:
            if x not in visi:
                visi.append(x)
        return visi


g = Graph(number, nDe)
g.Kruskal()
g.oddFinder()
g.perfectMatching()
g.EulerianTreeFinder()

E = Eulertour(g.vertices, sorted(g.EuTree,key=lambda item:item[2]), np.random.randint(0, number-1))
e = copy.deepcopy(E.getTour())

def tourCost(path):
    cost = 0
    for i in range(number):
        cost += distances[path[i]][path[(i+1)%number]]
    return cost

Visited = np.array(e)
start = time.time()
end = time.time()
curr = Visited
neighbours=0
while (time.time() - start < 50):
        p = random.random()
        # p = 0.4
        del neighbours
        neighbours = copy.deepcopy(curr)
        x = np.random.randint(0, number)
        y = np.random.randint(0, number)
        z = np.random.randint(0, number)
        m = min(x, y, z)
        M = max(x, y, z)
        if p <= 0.3:  # 2 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = temp
        elif p <= 0.6:  # 3 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = neighbours[z]
            neighbours[z] = temp
        else:  # insert
            temp = neighbours[m]
            for i in range(m, M):
                neighbours[i] = neighbours[i+1]
            neighbours[M] = temp
        if (tourCost(neighbours) -  tourCost(curr)) < 0:
            curr = neighbours
            print(tourCost(curr))
            for x in neighbours:
                pheromones[x][(x+1)%number]+=0.1

print(pheromones)
# exit(0)
numberOfAnts = 30
if number==100 and name=="euclidean":
    alpha = 3
    beta = 3
    rho = 0.1
    Q = 0.1
else:
    alpha = 40
    beta = 40
    rho = 0.1
    Q = 0.1
bestCost = float('Inf')
bestPath = []




# Ant Colony Optimization
try:
    while time.time() - start < 100:
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
        top = int(0.2*len(allAntsPath)+1)
        for path in allAntsPath[:top]:
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

try:
    print("Starting 2-Opt, 3-Opt and insert")
    Visited = np.array(bestPath)
    start = time.time()
    end = time.time()
    curr = Visited
    neighbours=0
    while (time.time() - start < 200):
        p = random.random()
        # p = 0.4
        del neighbours
        neighbours = copy.deepcopy(curr)
        x = np.random.randint(0, number)
        y = np.random.randint(0, number)
        z = np.random.randint(0, number)
        m = min(x, y, z)
        M = max(x, y, z)
        if p <= 0.3:  # 2 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = temp
        elif p <= 0.6:  # 3 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = neighbours[z]
            neighbours[z] = temp
        else:  # insert
            temp = neighbours[m]
            for i in range(m, M):
                neighbours[i] = neighbours[i+1]
            neighbours[M] = temp
        if (tourCost(neighbours) -  tourCost(curr)) < 0:
            curr = neighbours
            print(tourCost(curr))
            # print (list(curr))
        end = time.time()
except KeyboardInterrupt:
    print("Interrupted on user demand.")
    # print(tourCost(curr), end-start)
# print(*curr)
print ("Best Cost : " ,tourCost(curr))
print (list(curr))
