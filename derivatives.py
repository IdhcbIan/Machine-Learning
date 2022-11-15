import random
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 3*x**2 - 4*x + 5



"""
#---// Making a data set for a graph //-------

l = []
for i in range(-101, 101):
    l.append(f(i))

print(l)
#-----// Making the graph //------------

plt.plot(l)
plt.show() 



#-------// Or in pro mode //------------
xs = np.arange(-5, 5, 0.25)     # Learn arrays!!!!
ys = f(xs)

print(ys)
plt.plot(xs, ys)
plt.show()

"""
# example of derivartive by definition

"""
h = 0.00001
x = 2/3
print((f(x+h) - f(x))/h)
"""

# Getiting more complex

"""
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)
"""
h = 0.00001

# inputs
a = 2.0
b = -3.0
c = 10.0


d1 = a*b + c
a += h
d2 = a*b + c

print("d1", d1)
print("d2", d2)
print("slope", (d2 - d1)/h)