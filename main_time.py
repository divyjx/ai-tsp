import time

start = time.time()
import numpy as np
import sys 
from numpy import random
import math
# f=open(sys.argv[1],"r")
f=open("euc_100","r")
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
global nD
# print(nC)
disMat=[]
for i in range(number):
    x=f.readline().rstrip("\n").split()
    disMat.append(x)
nD=np.array(list(map(conv,disMat)))
# print(nD,nC)

#1 naive approach - minimum sussesive roots

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
# node=[for i in range(100)]
node=[i for i in range(100)]

def costEval(node):
    cost=0
    for x in range(len(node)):
        From=node[x]
        if x==99:
            To=node[0]
        else:
            To=node[x+1]
        cost+=nD[From][To]
    return cost
    
def randomNeighbour():
    p=random.permutation(np.array([i for i in range (100)]))
    # print (p)
    return p

def Cooling(temp,time):
    return temp*(0.9**time)

def simAn():
    Node=randomNeighbour()
    bestNode=Node
    print(costEval(bestNode))
    Temp=1000000
    M=200 # while loop termination
    Epoch=10 # number of iterations at a temprature 
    # Time=0
    for Time in range(1,Epoch):
        i=0
        while i!=M:
            neighbours=randomNeighbour()
            if (costEval(neighbours)<8000):
                        bestNode=neighbours
                        break
            delE=costEval(neighbours)-costEval(Node)
            exp=1/(1+pow(math.e,(-delE/Temp)))
            # if (exp<0.0010000000000001):
            #     # return (costEval(bestNode),costEval(Node))
            #     exp=0.001
            print(exp)
            if random.random()<exp:
                Node=neighbours
                if costEval(bestNode)>costEval(Node):
                    bestNode=Node
                    if (costEval(bestNode)<8000):
                        break    
                    # print(Temp,costEval(bestNode))
                Temp=Cooling(Temp,Time)
            i+=1
    return costEval(bestNode)


print(simAn())

end = time.time()
print(end - start)