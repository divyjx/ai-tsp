import numpy as np
import sys 
# from numpy import random
import random
import math
import copy
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

tour = np.random.permutation(np.array([i for i in range (number)]))

def findCost(tour):
    cost=0
    for i in range(len(tour)):
        From=tour[i]
        To = tour[(i+1)%number]
        cost += nD[From][To]
    return cost

def leftRightCost(tour, left, right):
    left = min(left, right)
    right = max(left, right)
    # if left + 1 != right:
    cost = nD[tour[left-1]][tour[left]] + nD[tour[left]][tour[(left+1)%number]] + nD[tour[right-1]][tour[right]] + nD[tour[right]][tour[(right+1)%number]]
      

    return cost

def simulatedAnnealing():
    tour_best = tour[:]
    cost_best = findCost(tour_best)
    sc ,rc ,ic = 0,0,0
    best_costs = []

    temp_start = 1.0e+500
    temp_end = 1e-20
    cooling_factor = 0.99
    try:
        for iteration in range(1):
            temp = temp_start
            tour_curr = tour_best[:]
            cost_curr = cost_best
            tour_new = tour_best[:]
            cost_new = cost_best

            step = 0
            while temp > temp_end:
                # select cities te be swapped, (no need to swap first city)
                

                # neighbhour funtion
                p = random.random()
                # p=0.9
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                m = min(x, y)
                M = max(x, y)
                # # neighbours = copy.deepcopy(tour_curr)
                if p <= 0.5:  # swap
                    tour_new[x], tour_new[y]= tour_new[y],tour_new[x]
                    sc+=1
                elif 0.5 < p and p <= 0.8:  # reverse
                    while m > M:
                        tempv = tour_new[m]
                        tour_new[m] = tour_new[M]
                        tour_new[M] = tempv
                        M -= 1
                        m += 1
                    rc+=1
                else:  # insert
                    tempv = tour_new[m]
                    for i in range(m, M):
                        tour_new[i] = tour_new[i+1]
                    tour_new[M] = tempv
                    ic+=1
                cost_new=findCost(tour_new)

                # tour_new=copy.deepcopy(neighbours)
                
                # index = random.sample(range(number-1), 2)
                # index[0] += 1
                # index[1] += 1
                # before_swap = findCost(tour_new)
                # tour_new[index[0]], tour_new[index[1]] = tour_new[index[1]], tour_new[index[0]]
                # after_swap = findCost(tour_new)
                # cost_new = after_swap
                
                # print(cost_new)
                
                delE = cost_new - cost_curr
                if delE < 0 or 1/(1+math.exp(-delE/temp)) > random.random():
                    tour_curr = tour_new[:]
                    cost_curr = cost_new
                else:
                    # reset
                    tour_new = tour_curr[:]
                    cost_new = cost_curr

                if cost_best > cost_curr:
                    tour_best = tour_curr[:]
                    cost_best = cost_curr
                
                if step % 100 == 0:
                    best_costs.append(cost_best)

                temp = temp * cooling_factor
                step += 1
    except KeyboardInterrupt as e:
        print ("Interrupted on user demand.",sc,rc ,ic)
    return cost_best


print((simulatedAnnealing()))
# print(leftRightCost(tour, 2, 55))
# print(findCost(tour))