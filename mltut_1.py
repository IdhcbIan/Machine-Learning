import random
import numpy as np
import matplotlib.pyplot as plt
import math

y = []
x = []

for i in range(0, 100):
    y.append(round(random.random()*2, 2))



print(y)
l = []
for i in y:
    l.append(round(1/(1 + (math.e ** -i)), 2))

print(l)

plt.plot(l)
plt.show() 

