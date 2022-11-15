# This is some code that i used to learn backpropagation
# and i used it to find the minimum of functions


import random
import numpy as np
import matplotlib.pyplot as plt
from micrograd.engine import Value

# Variables 
g = -1.0   # The derivative, that we cant to make 0, local minimum
c = float(input()) # The random point that we will start, in a way this is x  
nudge = 0 # the nudge of x depending on the derivative

win = 0  # If we found the local minimum

# This is our Function...

def f(x):
    return (x)**2 -1  # it will find the minimum of any function

# this is the Backpropagation

def cauculate(nudge, c, g): # It gets in the Nudge, the point and the derivative
    a = Value(c + nudge)
    F = f(a)
    F.backward()
    g = (a.grad) # Df/Da
    c = c + nudge

    if g == 0:
        print("this is the value of the minimum:", F.data)
    
    return c, g,

cauculate(nudge, c, g)

while win == 0:
    # Find the nudge depending in the derivative
    if g < 0:
        nudge = 1
    if g > 0:
        nudge = -1

    # Check is derivative is 0
    if g == 0:
        win = 1
        #print("nice")
        print("This is the x of the minimum:", c)

    c, g = cauculate(nudge, c, g)

