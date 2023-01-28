import sys
import math
from numpy import random
import numpy as np
import time
import copy

start = time.time()
# f=open(sys.argv[1],"r")
f = open("eu_matrix.txt", "r")
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
# print(nD,nC)

# 1 naive approach - minimum sussesive roots

# global visited
# visited=[]
# start=9
# startCorr=nC[0]
# cost=0
# curr=50
# def mini(curr):
#     m=100000
#     ind=0
#     for i in range(100):
#         if i in visited:
#             continue
#         if nD[curr][i]<m:
#             m=nD[curr][i]
#             ind=i
#     return m,ind
# i=0
# while len(visited)!=5:
#     if curr not in visited:
#         visited.append(curr)
#         x=mini(curr)
#         cost+=x[0]
#         curr=x[1]
#     i+=1
# # print(curr,cost)
# cost+=(nD[visited[-1]][0])
# print(visited,cost) ## cost =~ 100000


# 2 - brute force

# 3 - mst and preorder approach

# 4 Stimulated annealing - page 103

def costEval(node):
    # print(node)
    cost = 0
    for x in range(len(node)):
        From = node[x]
        if x == 99:
            To = node[0]
        else:
            To = node[x+1]
        cost += nD[From][To]
    return cost


def randomNeighbour(Node):
    p = random.permutation(np.array([i for i in range(100)]))
    return p


def Cooling(temp, time):
    # return temp*(0.9999*time)
    return temp*(0.999)
    # return temp*(0.99)


def simAn(node):
    Node = randomNeighbour(node)
    bestNode = Node
    print(costEval(bestNode))
    Temp = 2000000
    M = 100  # while loop termination
    Epoch = 20000  # number of iterations at a temprature
    Time = 1
    time = 1
    # for Time in range(1, Epoch):
    while Time != Epoch:
        try:
            # i = 0
            # while i != M:
            # while Temp > M:

            p = random.random()
            if p <= 0.001:
                neighbours = randomNeighbour(Node)
            elif 0.001 < p and p <= 0.5:  # swap
                neighbours = copy.deepcopy(Node)
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                temp = neighbours[x]
                neighbours[x] = neighbours[y]
                neighbours[y] = temp
            elif 0.5 < p and p <= 0.8:  # reverse
                neighbours = copy.deepcopy(Node)
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                m = min(x, y)
                M = max(x, y)
                while m > M:
                    temp = neighbours[m]
                    neighbours[m] = neighbours[M]
                    neighbours[M] = temp
                    M -= 1
                    m += 1
            else:  # insert
                neighbours = copy.deepcopy(Node)
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                m = min(x, y)
                M = max(x, y)
                temp = neighbours[m]
                # if M!=m:
                for i in range(m, M):
                    neighbours[i] = neighbours[i+1]
                neighbours[M] = temp

            delE = costEval(neighbours)-costEval(Node)
            if delE < 0:
                exp = 1/(1+(pow(math.e, (-delE/Temp))))
                # exp = (pow(math.e, (-delE/Temp)))
                print(exp)
            # print(delE)
            else:
                exp = 1
            if random.random() < exp:
                Node = neighbours
                if costEval(bestNode) > costEval(Node):
                    bestNode = Node
            Temp = Cooling(Temp, Time)
                # print(Temp)
            Time += 1

            # i += 1
        except KeyboardInterrupt:
            return costEval(bestNode), exp, delE, Temp, Time
            break

        else:
            continue
    return costEval(bestNode), exp, delE, Temp, Time


node = np.random.permutation([i for i in range(100)])
# print(simAn(node))



end = time.time()
print(end - start)


# gradient desent algorithm in sa
