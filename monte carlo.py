# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:23:33 2025

@author: Admin
"""
 
import numpy as np
import matplotlib.pyplot as plt

# Parameters for integration
a = 0
b = np.pi  # Limits of integration
N = 1000  # Number of random samples per estimate
iterations = 100  # Number of times to repeat the estimation (for better distribution)
xrand = np.random.rand(N) 
areas = []  # Initialize areas list to store multiple results

def func(x):
    return np.sin(x)

# Run Monte Carlo integration multiple times
for _ in range(iterations):
    integral = 0.0

    for i in range(N):
        x = a + (b - a) * xrand[i]  # Scale the random number to the range [a, b]
        integral += func(x)

    # Estimate the integral
    answer = (b - a) / float(N) * integral
    areas.append(answer)  # Append each estimate to the areas list

# Plot the distribution of areas
plt.title("Distribution of Areas")
plt.hist(areas, bins=30, edgecolor='black')  # Show distribution of results
plt.xlabel("Areas")
plt.ylabel("Frequency")
plt.show()


