from collections import defaultdict
import copy
import numpy as np
import sys
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


def tourCost(node):
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

g = Graph(number, nDe)
g.Kruskal()
g.oddFinder()
g.perfectMatching()
g.EulerianTreeFinder()
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



E = Eulertour(g.vertices, sorted(g.EuTree,key=lambda item:item[2]), np.random.randint(0, number-1))
e = copy.deepcopy(E.getTour())

bestCost=tourCost(e)
bestPath=e

try:
    print("Starting 2-Opt and 3-Opt")
    curr = np.array(bestPath)
    currCost = bestCost
    neighbours = 0
    p=0.1
    while (time.time() - start < 300):
        p = random.random()
        # p = 0.1
        del neighbours
        neighbours = copy.deepcopy(curr)
        while (True):
            r = sorted(random.choices([i for i in range(number-1)], k=3))
            x, y, z = r
            if (y-x) >= 2 and (z-y) >= 2 and (x-z+number) % number >= 2:
                break

        m = min(r)
        M = max(r)

        if p < .2:  # 2 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = temp
        elif p < .4:  # 3 swap
            temp = neighbours[x]
            neighbours[x] = neighbours[y]
            neighbours[y] = neighbours[z]
            neighbours[z] = temp
        elif p < .6:  # 2-opt
            neighbours[m:M] = neighbours[m:M][::-1]
        elif p < .8:  # 3-opt
            neighbours = list(neighbours)
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
            neighbours = np.array(neighbours)

        else:  # insert
            temp = neighbours[m]
            for i in range(m, M):
                neighbours[i] = neighbours[i+1]
            neighbours[M] = temp
        if (tourCost(neighbours) - currCost) < 0:
            currCost = tourCost(neighbours)
            curr = neighbours
            print(currCost)
            # print(list(neighbours))
        # p+=0.2
        # p=p%1
except KeyboardInterrupt:
    print(tourCost(curr))
print("Best Cost : ", currCost)
print(list(curr))


    ######      run mst_chris.py file first    #######
    ######    res list stores all euler edges  #######
    ###### euler_tree_edges.txt contains edges #######