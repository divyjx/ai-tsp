import numpy as np
import sys 
f=open(sys.argv[1],"r")
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
print(nD)
