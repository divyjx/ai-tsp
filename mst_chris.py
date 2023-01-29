from collections import defaultdict
import copy
import numpy as np
import sys
import math
from numpy import random
import numpy as np
import time
import copy

start = time.time()
# f=open(sys.argv[1],"r")
f = open("euc_100", "r")
Cities = []
name = f.readline().rstrip("\n")
number = int(f.readline().rstrip("\n"))

for i in range(number):
    x = f.readline().rstrip("\n").split()
    Cities.append(x)
# print(Cities)


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


class Graph():
    # self.vertices
    # edgeMatrix
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
        # print(self.edges)
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
        # print(self.PMtree)
        # if len(self.PMtree)==len(self.oddVertices)/2:
        #     print("yes") --->> working
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


g = Graph(100, nDe)
g.Kruskal()
g.oddFinder()
g.perfectMatching()
g.EulerianTreeFinder()

f=open("eu_tree_edges.txt","w")
res = copy.deepcopy(g.EuTree)
for x in res:
    for y in x :
        f.write(str(y) + " ")
    f.write ("\n")

# dfs caculation


# adj = defaultdict(list)
# vis = defaultdict(bool)
# Euler = [0]*(2*g.vertices)

# def add_edge(u, v, w):
#     adj[u].append((v, w))
#     adj[v].append((u, w))

# def eulerTree(u, index):
#     vis[u] = True
#     Euler[index] = u
#     index += 1
#     for nbr in adj[u]:
#         if not vis[nbr[0]]:
#             index = eulerTree(nbr[0], index)
#             Euler[index] = u
#             index += 1
#     return index

# def EulerTour(root, N):
#     path = []
#     index = 0
#     eulerTree(root, index)
#     for i in range(2*N-1):
#         path.append(Euler[i])
#     return path

# for x in g.EuTree:
#     add_edge(x[0], x[1], x[2])

# ETP = EulerTour(1, g.vertices)
# Visited = []

# for x in ETP:
#     if x not in Visited:
#         Visited.append(x)


# def costEval(node):
#     cost = 0
#     for x in range(len(node)):
#         From = node[x]
#         if x == 99:
#             To = node[0]
#         else:
#             To = node[x+1]
#         cost += nD[From][To]
#     return cost


# Visited = np.array(Visited)
# end = time.time()
# curr = Visited
# try:
#     while (end - start < 200):

#         p = random.random()
#         p=0.4
#         neighbours = copy.deepcopy(curr)
#         x = np.random.randint(0, g.vertices)
#         y = np.random.randint(0, g.vertices)
#         z = np.random.randint(0, g.vertices)
#         m = min(x, y,z)
#         M = max(x, y,z)
#         if p <= 0.5:  # swap
#             temp = neighbours[x]
#             neighbours[x] = neighbours[y]
#             neighbours[y] = neighbours[z]
#             neighbours[z] = temp
#         else:  # insert
#             temp = neighbours[m]
#             for i in range(m, M):
#                 neighbours[i] = neighbours[i+1]
#             neighbours[M] = temp
#         if costEval(neighbours) < costEval(curr):
#             curr = neighbours
#             print(costEval(curr))
#         end = time.time()
# except KeyboardInterrupt:
#     print(costEval(curr), end-start)
# print(*curr)
## print(costEval(Visited))


######      run mst_chris.py file first    #######
######    res list stores all euler edges  #######
###### euler_tree_edges.txt contains edges #######
