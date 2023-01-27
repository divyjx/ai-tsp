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
# print(nC)
disMat=[]
for i in range(number):
    x=f.readline().rstrip("\n").split()
    disMat.append(x)
nD=np.array(list(map(conv,disMat)))
# print(nD,nC)

# naive approach - minimum sussesive roots
global visited
visited=[]
start=0
startCorr=nC[0]
cost=0
curr=0
def mini(arr):
    m=100000
    ind=0
    for i in range(len(arr)):
        if i in visited:
            continue
        if arr[i]<m:
            m=arr[i]
            ind=i
    return m,i        
while len(visited)!=100:
    visited.append[curr]
    x=mini(nD[curr])
    cost+=mini[0]
    curr=mini[1]
print(curr,cost)
