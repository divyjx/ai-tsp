from collections import defaultdict
import copy
import numpy as np
import sys
import math
from numpy import random
import time
import random
start = time.time()
f=open(sys.argv[1],"r")
# f = open("noneuc_500", "r")
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
global nD
# print(nC)
disMat = []
for i in range(number):
    x = f.readline().rstrip("\n").split()
    disMat.append(x)
nD = np.array(list(map(conv, disMat)))

nDe = []
for x in range(len(nD)):
    for y in range(x+1, len(nD)):
        nDe.append([x, y, nD[x, y]])


def costEval(node):
    cost = 0
    for x in range(number):
        From = node[x]
        if x == (number-1):
            To = node[0]
        else:
            To = node[x+1]
        cost += nD[From][To]
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
# e=open("eu_tree_edges.txt","r")
# new=[]
# for i in range(number):
#     x=list(map(int,e.readline().rstrip("\n").split()))
#     new.append(x)

# # for x in range(number):
# #     for y in range(x,number):
# #         if [x,y] not in new: 
# #             nDe[x][y]+=1000
# for i in range(len(nDe)):
#     if [nDe[i][0],nDe[i][1]] not in new:
#         nDe[i][2]+=1000

g = Graph(number, nDe)
g.Kruskal()
g.oddFinder()
g.perfectMatching()
g.EulerianTreeFinder()

# # g.EuTree -->> eulermultigraph or tree

# # uncomment for ant colony opt with increased pheromones
#
f=open("eu_tree_edges.txt","w")
res = copy.deepcopy(g.EuTree)
for x in res:
    for y in range(2) :
        f.write(str(x[y]) + " ")
    f.write ("\n")
print("here")
if __name__=="__main__":
# exit(0)
    print("not here")
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



    # mini=100000000 
    # end=time.time()
    # while(end-start < 100 ):
    #     per = np.random.permutation(g.EuTree)
    #     E = Eulertour(g.vertices, per, np.random.randint(0, number))
    #     e = copy.deepcopy(E.getTour())
    #     del E,per
    #     c=costEval(e)
    #     if c<mini:
    #         mini=c 
    #     end=time.time()    
    # print(mini)

    E = Eulertour(g.vertices, sorted(g.EuTree,key=lambda item:item[2]), np.random.randint(0, number-1))
    e = copy.deepcopy(E.getTour())
    # while(1):
    #     freq=[0 for i in range(number)]
    #     paths=[]
    #     for x in E.path:
    #         x=int (x)
    #         paths.append(x)
    #         freq[x]+=1
    #     # paths=np.random.permutation(paths)
    #     for x in range(number):
    #         freq[x]-=1

    #     while(len(paths)!=number):

    #         wh=[i/sum(freq) for i in freq]
    #                 # J = random.choices(remainingCities, weights=probList_ij)[0] # highest prob is 0th element 
    #         x=random.choices([i for i in range(number)],weights=wh)[0]
    #         # print(x)
    #         # exit()
    #         freq[x]-=1
    #         paths=list(paths)
    #         paths.remove(x)

        # print(costEval(paths))
    # exit(0)
    # curr_cost=20000
    # while True:
    #     path=copy.deepcopy(E.path)
    #     while len(path)!=number:
    #         p=np.random.randint(0,len(path))
    #         path.pop(p)
    #         # print(path.pop(p),end=" ")
    #     # print("")
    #     visi = []
    #     for x in path:
    #         if x not in visi:
    #             visi.append(x)
    #     if len(visi)==number:
    #         if(costEval(visi))<curr_cost:
    #             curr_cost=costEval(visi)
    #         print(curr_cost)

    try:
        Visited = np.array(e)
        start = time.time()
        end = time.time()
        curr = Visited
        neighbours=0
        while (end - start < 300):
            p = random.random()
            # p = 0.4
            del neighbours
            neighbours = copy.deepcopy(curr)
            x = np.random.randint(0, g.vertices-1)
            y = np.random.randint(0, g.vertices-1)
            z = np.random.randint(0, g.vertices-1)
            a = np.random.randint(0, g.vertices-1)
            b = np.random.randint(0, g.vertices-1)
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
            # elif p <= 0.9:  # 3 swap
            #     temp = neighbours[x]
            #     neighbours[x] = neighbours[y]
            #     neighbours[y] = neighbours[z]
            #     neighbours[z] = neighbours[a]
            #     neighbours[a] = neighbours[b]
            #     neighbours[b] = temp
            else:  # insert
                temp = neighbours[m]
                for i in range(m, M):
                    neighbours[i] = neighbours[i+1]
                neighbours[M] = temp
            # if (costEval(neighbours) - costEval(curr)) < (-0.1*number):
            if (costEval(neighbours) -  costEval(curr)) < 0:
                curr = neighbours
                print(costEval(curr))
            end = time.time()
    except KeyboardInterrupt:
        print(costEval(curr), end-start)
    # print(*curr)
    path=curr


    ######      run mst_chris.py file first    #######
    ######    res list stores all euler edges  #######
    ###### euler_tree_edges.txt contains edges #######

import numpy as np
import random
import math

def euclidean_distance(city1, city2):
    x_distance = abs(city1[0] - city2[0])
    y_distance = abs(city1[1] - city2[1])
    distance = np.sqrt((x_distance ** 2) + (y_distance ** 2))
    return distance

def calculate_distance(city_list):
    # distance = 0
    # for i in range(len(city_list)):
    #     j = (i + 1) % len(city_list)
    #     distance += euclidean_distance(city_list[i], city_list[j])
    return costEval(city_list)

def swap_cities(city_list, index1, index2):
    new_city_list = city_list.copy()
    new_city_list[index1], new_city_list[index2] = new_city_list[index2], new_city_list[index1]
    return new_city_list

def simulated_annealing(city_list, temperature, cooling_rate, termination_temp):
    current_distance = calculate_distance(city_list)
    best_distance = current_distance
    best_solution = city_list
    
    while temperature > termination_temp:
        index1, index2 = random.sample(range(len(city_list)), 2)
        new_solution = swap_cities(city_list, index1, index2)
        new_distance = calculate_distance(new_solution)
        delta_distance = new_distance - current_distance
        if delta_distance < 0:
            city_list = new_solution
            current_distance = new_distance
            if current_distance < best_distance:
                print(current_distance)
                best_distance = current_distance
                best_solution = city_list
        else:
            acceptance_probability = math.exp(-delta_distance / temperature)
            if acceptance_probability > random.uniform(0, 1):
                print(current_distance)
                city_list = new_solution
                current_distance = new_distance
        temperature *= cooling_rate
    return best_solution

city_list = [i for i in range (number)] # Example cities
temperature = 10 # Starting temperature
cooling_rate = 0.9999 # Cooling rate
termination_temp = 1e-16 # Termination temperature

best_solution = simulated_annealing(path, temperature, cooling_rate, termination_temp)
print("Best solution:", best_solution)
print("Best solution:", costEval(best_solution))
