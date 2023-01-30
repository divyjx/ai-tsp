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
        # print(len(self.PMtree),len(self.oddVertices))
        # exit(0)
        return (self.PMtree)

    def EulerianTreeFinder(self):
        self.EuTree = copy.deepcopy(self.KruskalTreeEdges)
        for x in self.PMtree:
            if x not in self.EuTree:
                self.EuTree.append(x)
        # print(len(self.KruskalTreeEdges))
        # print(self.PMtree,"\n")
        # print(self.EuTree)
        # exit(0)
        self.EuMatrix = np.zeros(shape=(self.vertices, self.vertices))
        for x in self.EuTree:
            self.EuMatrix[x[0]][x[1]] = x[2]
            self.EuMatrix[x[1]][x[0]] = x[2]

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

# exit(0)
# class Eulertour:
#     def __init__(self, vertices, Edges, root) -> None :
#         self.adj = defaultdict(list)
#         self.vis = defaultdict(bool)
#         self.vertices = vertices
#         self.path = []
#         self.edges = copy.deepcopy(Edges)
#         self.addEdge()
#         self.Euler = [0]*(2*vertices)
#         self.root = root

#     def addEdge(self):
#         self.adj.clear
#         self.vis.clear
#         for u, v, w in self.edges:
#             self.adj[u].append(v)
#             self.adj[v].append(u)

#     def Tour(self, start, index):
#         self.vis[start] = True
#         self.Euler[index] = start
#         index += 1
#         for x in self.adj[start]:
#             if not self.vis[x]:
#                 index = self.Tour(x, index)
#                 self.Euler[index] = x
#                 index += 1
#         return index

#     def getTour(self):
#         self.addEdge()
#         path = []
#         index = 0
#         self.Tour(self.root, index)
#         for i in range(2*self.vertices):
#             path.append(int(self.Euler[i]))
#         visi = []
#         self.path=path
#         for x in path:
#             if x not in visi:
#                 visi.append(x)
#         return visi



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

# E = Eulertour(g.vertices, sorted(g.EuTree,key=lambda item:item[2]), np.random.randint(0, number))
# e = copy.deepcopy(E.getTour())
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

# try:
#     Visited = np.array(e)
#     start = time.time()
#     end = time.time()
#     curr = Visited
#     neighbours=0
#     while (end - start < 300):
#         p = random.random()
#         # p = 0.9
#         del neighbours
#         neighbours = copy.deepcopy(curr)
#         x = np.random.randint(0, g.vertices)
#         y = np.random.randint(0, g.vertices)
#         z = np.random.randint(0, g.vertices)
#         a = np.random.randint(0, g.vertices)
#         b = np.random.randint(0, g.vertices)
#         m = min(x, y, z)
#         M = max(x, y, z)
#         if p <= 0.3:  # 2 swap
#             temp = neighbours[x]
#             neighbours[x] = neighbours[y]
#             neighbours[y] = temp
#         elif p <= 0.6:  # 3 swap
#             temp = neighbours[x]
#             neighbours[x] = neighbours[y]
#             neighbours[y] = neighbours[z]
#             neighbours[z] = temp
#         # elif p <= 0.9:  # 3 swap
#         #     temp = neighbours[x]
#         #     neighbours[x] = neighbours[y]
#         #     neighbours[y] = neighbours[z]
#         #     neighbours[z] = neighbours[a]
#         #     neighbours[a] = neighbours[b]
#         #     neighbours[b] = temp
#         else:  # insert
#             temp = neighbours[m]
#             for i in range(m, M):
#                 neighbours[i] = neighbours[i+1]
#             neighbours[M] = temp
#         # if (costEval(neighbours) - costEval(curr)) < (-0.1*number):
#         if (costEval(neighbours) -  costEval(curr)) < 0:
#             curr = neighbours
#             print(costEval(curr))
#         end = time.time()
# except KeyboardInterrupt:
#     print(costEval(curr), end-start)
# # print(*curr)
# path=curr


######      run mst_chris.py file first    #######
######    res list stores all euler edges  #######
###### euler_tree_edges.txt contains edges #######


# print(len(g.EuTree))
Subgraph = copy.deepcopy(g.EuTree)

class Problem():
    def __init__(self,edges,numAnts, alpha, beta, rho, Q) -> None:
        # numbering of edges acc to weights
        self.edges=sorted(edges,key=lambda item:item[2])
        self.numEdges=len(edges)
        self.numAnts=numAnts
        self.Q = Q
        self.beta = beta
        self.rho = rho
        self.alpha = alpha
        self.pheromones=[0.5 for x in range(len(edges))]
        self.bestCost = float('Inf')
        self.bestTour = range(number)

    def ACO(self):
        try:
            while time.time()-start < 10:
                antsList=[]
                pheromone_change=[0 for i in range(self.numEdges)]

                for i in range(self.numAnts):
                    ant=AntObject(self.edges,self.pheromones,self.alpha,self.beta)
                    antsList.append(ant)

                    if ant.pathCost() < self.bestCost:
                        self.bestCost = ant.pathCost()
                        self.bestTour=ant.currentPath
                        print(f"Current Best Cost: {self.bestCost}")
                        # print(*self.bestTour, end="\n\n")
                
                antsList.sort(key=lambda ant: ant.pathCost())

                for ant in antsList:
                    currPath=ant.currentPath
                    edgeIndexes=[]
                    for i in range(len(currPath)):
                        try:
                            self.edges.index([currPath[i],currPath[(i+1)%number],any])
                        except ValueError:
                            edgeIndexes.append(self.edges.index([currPath[(i+1)%number],currPath[i],any]))
                        else:
                            edgeIndexes.append(self.edges.index([currPath[i],currPath[(i+1)%number],any]))
                    for j in edgeIndexes:
                        pheromone_change[j]+=self.Q/self.edges[j][2]

                for j in edgeIndexes:
                    self.pheromones[j]=(1-self.rho)*self.pheromones[j]+pheromone_change[j]
        except KeyboardInterrupt:
            print(self.bestCost)
            print(*self.bestTour)
            exit(0)


                        


class AntObject():
    def __init__(self,edges,pheromones,alpha,beta) -> None:
        self.currentPath = []
        self.formPath(edges,pheromones,alpha,beta)
        # self.pathCost=costEval(self.currentPath)
    def formPath(self,edges,pheromones,alpha,beta):
        # print(1)
        startCity=random.randint(0,number)
        remainingCities=list(range(0,number))
        remainingCities.remove(startCity)
        self.currentPath.append(startCity)
        # print(edges)    
        while(len(self.currentPath)<number):
            i = self.currentPath[-1]
            # print(i)
            edgeIndexes=[]
            # for x in range(len(edges)):
            #     if edges[x][0]==i:
            #         edgeIndexes.append(x)
            #     elif edges[x][1]==i:
            #         edgeIndexes.append(x)
            #     if len(edgeIndexes)!=0:
            #         if edgeIndexes[-1] in  remainingCities:
            #             edgeIndexes.pop()
            frm=i
            to=[]
            # print(len(edges))
            # exit(0)
            for x in range(len(edges)):
                if frm==edges[x][1]:
                    edgeIndexes.append(x)
                    if  edges[edgeIndexes[-1]][0] not in remainingCities:
                        edgeIndexes.pop()
                elif frm==edges[x][0]:
                    edgeIndexes.append(x) 
                    if  edges[edgeIndexes[-1]][1] not in remainingCities:
                        edgeIndexes.pop()
            # print(edgeIndexes)
            # print(edges[edgeIndexes[1]])
            # exit(0)
            # edgeIndexes=[]
            # for i in range(len(edges)):
            #     try:
            #         self.edges.index([i,any,any])
            #     except ValueError:
            #         edgeIndexes.append(self.edges.index([currPath[(i+1)%number],currPath[i],any]))
            #     else:
            #         edgeIndexes.append(self.edges.index([currPath[i],currPath[(i+1)%number],any]))    


            pheromonesAB=[(pheromones[k]**alpha * (1/edges[k][2])**beta) for k in edgeIndexes]
            pheromones_prob=[float(x/sum(pheromonesAB)) for x in pheromonesAB]
            # print(self.currentPath,edgeIndexes,pheromonesAB)
            # for e in edgeIndexes:
            #     print(edges[e])
            # print("")
            J=edges[random.choices(edgeIndexes,weights=pheromones_prob)[0]]
            if J[0]==i:
                self.currentPath.append(J[1])
                # print(J[1])
                # print(self.currentPath)
                remainingCities.remove(J[1])
            else:
                # print(J[0])
                # print(self.currentPath)
                self.currentPath.append(J[0])
                remainingCities.remove(J[0])

    def pathCost(self):
        return costEval(self.currentPath)            

# for x in g.OddTree:
#     if x not in Subgraph:
#         Subgraph.append(x)
# for x in g.edges:
#     if x not in Subgraph:
#         Subgraph.append(x)
problem = Problem(Subgraph, number, 5, 5, 0.75, 0.75)
# e=open("eu_tree_edges.txt","r")
# for i in range(number):
#     x=list(map(int,e.readline().rstrip("\n").split()))
#     problem.pheromones[x[0]][x[1]]+=.5
#     problem.pheromones[x[1]][x[0]]+=.5

problem.ACO()