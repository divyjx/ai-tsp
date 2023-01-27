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
    return 0.9 *temp
def simAn():
    Node=randomNeighbour()
    bestNode=Node
    print(costEval(bestNode))
    Temp=100000000
    M=100 # while loop termination
    i=0
    Epoch=200 # number of iterations at a temprature 
    # Time=0
    delE=0
    for Time in range(1,Epoch):
        while i!=M:
            neighbours=randomNeighbour()
            delE=costEval(neighbours)-costEval(Node)
            exp=1/(1+pow(math.e,(-delE/Temp)))
            print(exp)
            if random.random()<exp:
                Node=neighbours
                if costEval(bestNode)>costEval(Node):
                    bestNode=Node
                Temp=Cooling(Temp,Time)
            i+=1
    return costEval(bestNode)


# print(simAn())
# print(costEval([86,33,84,15,64,55,9,30,61,70,50,87,43,79,66,32,12,93,2,25,20,46,8,73,75,88,98,36,29,41,48,63,1,54,97,22,16,26,72,11,28,60,27,74,34,92,96,19,4,57,85,10,18,13,52,65,0,38,81,21,39,24,47,5,76,6,95,69,83,68,31,49,82,23,14,51,91,56,7,53,45,3,94,40,67,89,90,44,42,78,35,58,37,77,71,62,80,59,17,99])) -->>> 1621.01680 -->> minimum using tsp