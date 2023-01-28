import numpy as np
import sys 
from numpy import random
import math
# f=open(sys.argv[1],"r")
f=open("eu_matrix.txt","r")
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

global visited
visited=[]
start=9
startCorr=nC[0]
cost=0
curr=50
def mini(curr):
    m=100000
    ind=0
    for i in range(100):
        if i in visited:
            continue
        if nD[curr][i]<m:
            m=nD[curr][i]
            ind=i
    return m,ind       
i=0
while len(visited)!=5:
    if curr not in visited:
        visited.append(curr)
        x=mini(curr)
        cost+=x[0]
        curr=x[1]
    i+=1
# print(curr,cost)
cost+=(nD[visited[-1]][0])
print(visited,cost) ## cost =~ 100000


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
    
# def randomNeighbour():
#     p=random.permutation(np.array([i for i in range (100)]))
#     # print (p)
#     return p

# def Cooling(temp,time):
#     return 0.9 *temp
# def simAn():
#     Node=randomNeighbour()
#     bestNode=Node
#     print(costEval(bestNode))
#     Temp=100000000
#     M=100 # while loop termination
#     i=0
#     Epoch=200 # number of iterations at a temprature 
#     # Time=0
#     delE=0
#     for Time in range(1,Epoch):
#         while i!=M:
#             neighbours=randomNeighbour()
#             delE=costEval(neighbours)-costEval(Node)
#             exp=1/(1+pow(math.e,(-delE/Temp)))
#             print(exp)
#             if random.random()<exp:
#                 Node=neighbours
#                 if costEval(bestNode)>costEval(Node):
#                     bestNode=Node
#                 Temp=Cooling(Temp,Time)
#             i+=1
#     return costEval(bestNode)


# print(simAn())
# print(costEval([86,33,84,15,64,55,9,30,61,70,50,87,43,79,66,32,12,93,2,25,20,46,8,73,75,88,98,36,29,41,48,63,1,54,97,22,16,26,72,11,28,60,27,74,34,92,96,19,4,57,85,10,18,13,52,65,0,38,81,21,39,24,47,5,76,6,95,69,83,68,31,49,82,23,14,51,91,56,7,53,45,3,94,40,67,89,90,44,42,78,35,58,37,77,71,62,80,59,17,99])) -->>> 1621.01680 -->> minimum using tsp
# print(costEval([51,14,23,31,68,69,83,94,80,0,62,70,79,12,8,66,93,41,26,22,72,97,10,18,57,29,88,75,73,46,32,98,48,16,63,1,28,11,81,76,5,47,6,95,49,53,82,78,40,67,89,45,77,43,58,44,42,35,37,87,30,50,61,9,55,64,36,52,13,91,56,65,19,4,74,38,96,39,21,24,92,34,27,60,86,33,85,15,84,99,17,59,54,20,25,2,71,90,3,7])) #-->>> 3619.01680 -->> minimum using genetic
# print(costEval([68,31,14,49,91,51,7,56,23,82,94,40,3,80,71,58,77,15,18,65,13,86,52,42,44,90,67,89,78,35,37,99,53,45,17,62,33,84,64,9,55,30,59,61,87,43,50,70,79,32,66,12,2,93,25,88,20,46,8,75,73,36,29,98,63,41,48,57,97,54,1,16,26,11,28,72,22,60,74,27,96,38,0,19,4,85,10,92,34,21,5,95,47,81,39,24,6,76,69,83])) #-->>> 2473.01620 -->> minimum using modified tsp
# print(costEval([22,26,16,97,1,63,48,29,36,98,41,54,10,85,57,4,19,38,0,18,15,64,55,33,84,52,13,86,99,17,59,77,30,9,25,20,46,88,75,73,8,32,66,12,2,93,79,70,50,43,87,61,37,58,78,35,89,67,40,3,94,90,42,44,71,62,80,45,53,23,49,68,83,69,31,91,51,14,82,7,56,65,39,24,5,76,6,95,47,21,81,34,92,96,27,74,60,11,28,72])) #-->>> 1908.01620 -->> minimum using modified pso
# print(costEval([1,63,48,54,48,63,41,29,36,29,98,88,20,25,12,32,66,32,46,8,73,75,35,89,67,40,94,3,82,23,14,91,51,91,56,7,53,45,80,62,71,77,37,58,78,58,37,77,71,62,44,42,90,42,44,62,80,99,17,59,17,99,33,84,15,64,55,9,30,61,70,50,87,43,87,50,70,79,70,61,30,93,2,93,30,9,55,64,15,84,33,86,52,13,18,10,85,57,4,19,60,27,74,96,92,34,92,81,21,39,24,39,21,47,5,76,6,76,5,95,69,83,68,31,49,31,68,83,69,95,5,47,21,81,92,96,74,27,60,19,4,57,97,22,72,11,28,26,16,26,28,11,72,22,97,57,85,0,38,65,38,0,85,10,18,13,52,86,33,99,80,45,53,7,56,91,14,23,82,3,94,40,67,89,35,75,73,8,46,32,2,25,20,88,98,29,41,63,1])) #-->>> 1908.01620 -->> minimum using modified pso

