import numpy as np
import sys 
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


