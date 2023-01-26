import numpy as np
# Code to Measure time taken by program to execute.
import time

# store starting time
begin = time.time()

# program body starts


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
print(nC)
disMat=[]
for i in range(number):
    x=f.readline().rstrip("\n").split()
    disMat.append(x)
nD=np.array(list(map(conv,disMat)))
print(nD)


# program body ends

time.sleep(1)
# store end time
end = time.time()

# total time taken
print(f"Total runtime of the program is {end - begin}")


#nC=city coordinates
#nD=city distance matrix