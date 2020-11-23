#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fernando Carrillo A01194204
25/11/20
2D Bin Packing Problem
using genetic algorithms
"""

from deap import base, creator, tools
from deap import algorithms
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

### PROBLEM VALUES ###
containerSize = (10, 10)
objects = [(5, 3), (3, 4), (1, 1), (2, 2), (2, 2), (4, 5), (3, 4), (4, 2), (5, 3), (2, 6),
        (4, 3), (2, 4), (2, 1), (2, 3), (6, 2), (3, 3), (4, 4), (6, 6), (6, 7), (3, 5),
        (3, 3), (2, 2), (3, 5), (4, 7), (8, 7), (5, 2), (3, 1), (1, 4), (2, 5), (5, 6),
        (3, 3), (4, 4), (2, 1), (3, 2), (4, 3), (1, 1), (3, 2), (5, 7), (5, 6), (5, 2),
        (3, 3), (4, 3), (2, 3), (1, 2), (6, 3), (2, 2), (3, 2), (1, 2), (5, 3), (2, 2),
        (4, 5), (3, 4), (4, 2), (5, 3), (2, 6), (4, 3), (2, 4), (2, 1), (2, 3), (6, 2),
        (3, 3), (4, 4), (6, 6), (6, 7), (3, 5), (3, 3), (2, 2), (3, 5), (4, 7), (4, 4),
        (5, 2), (3, 1), (1, 4), (2, 5), (5, 6), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3),
        (1, 1), (3, 2), (5, 7), (5, 6), (5, 2), (3, 3), (4, 3), (2, 3), (1, 2), (6, 3),
        (2, 2), (3, 2), (1, 2), (5, 3), (2, 2), (5, 3), (3, 4), (1, 1), (2, 2), (2, 2)]

### HELPER FUNCTIONS ###
"""
Fill in containers with objets starting by upper left corner.
1. Try to add object in the current row
2. If there's no space in row, create new row below it
3. If there's no space for a new row, create new container
Objects are placed as ordered
in the objects array.
(width, height)
"""
def FillContainers(container, objects):
    containersUsed = 1
    widthLeft = containerSize[0]
    heightLeft = containerSize[1]
    rowHeight = 0

    spaceWasted = 0
    objsInRow = []

    for obj in objects:
        # Adding object to current row
        if obj[0] <= widthLeft and obj[1] <= heightLeft:
            widthLeft -= obj[0]
            rowHeight = max(rowHeight, obj[1])
            objsInRow.append(obj)
        # Adding new row
        elif obj[0] > widthLeft and obj[1] <= (heightLeft - rowHeight):
            spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)
            objsInRow = [obj]

            widthLeft = containerSize[0] - obj[0]
            heightLeft -= rowHeight
            rowHeight = obj[1]
        # Adding new container
        else:
            spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)
            spaceWasted += (heightLeft-rowHeight) * containerSize[0]  # space left in container
            objsInRow = [obj]

            widthLeft = containerSize[0] - obj[0]
            heightLeft = containerSize[1]
            rowHeight = obj[1]
            containersUsed += 1
    spaceWasted += CalcSpaceWasted(containerSize[0], rowHeight, objsInRow)  # last obj in [objects]

    return (spaceWasted, containersUsed)

"""
Calculates the space wasted in a row by comparing the maximum height of an object
in the row with the rest of the objects, and the remaining width of row unused.
"""
def CalcSpaceWasted(width, height, objects):
    spaceWasted = 0
    widthLeft = width
    for obj in objects:
        spaceWasted += obj[0] * (height - obj[1])
        widthLeft -= obj[0]
    spaceWasted += widthLeft * height
    return spaceWasted

"""
Evaluation for individual that can calculate the number of containers
needed to fit all objects and the wasted space of a solution.
"""
def Evaluate(individual):
    global containerSize
    spaceWasted, containersUsed = FillContainers(containerSize, individual)
    return spaceWasted, containersUsed

"""
Mutates individual by alternating order of the objects in the array
"""
def MutSet(individual):
    newOrder = individual
    for i in range(len(objects) - 1):
        p = np.random.rand()
        if p < 0.25:
            newOrder[i], newOrder[i+1] = newOrder[i+1], newOrder[i]
    return newOrder,

"""
Mixes two individuals into a new one by copying the first 'pivot' elements
of ind1, and then trying to copy elements from ind2 if not used already.
"""
def MixInds(ind1, ind2, pivot):
    global objects
    notUsed = objects.copy()

    newInd = ind1[:pivot]  # grab first 'pivot' element from ind1
    # remove objects from notUsed used in newInd
    for i in range(len(newInd)):
        if newInd[i] in notUsed:
            del notUsed[notUsed.index(newInd[i])]

    # mate remaining objects with ind2 when possible
    for i in range(len(notUsed)):
        index = pivot + i
        if ind2[index] in notUsed:
            newInd.append(ind2[index])
            del notUsed[notUsed.index(ind2[index])]
        elif ind1[index] in notUsed:
            newInd.append(ind1[index])
            del notUsed[notUsed.index(ind1[index])]
        else:
            randInd = random.randint(0, len(notUsed)-1)
            newInd.append(notUsed[randInd])
            del notUsed[notUsed.index(notUsed[randInd])]

    return newInd

"""
Recieves two individuals and mates them by performing a crossover
from pivot places.
"""
def MateObjs(ind1, ind2, creator):
    pivot = int(random.randint(0, len(ind1)))
    newInd1 = MixInds(ind1, ind2, pivot)
    newInd2 = MixInds(ind2, ind1, pivot)
    
    return creator(newInd1), creator(newInd2)

"""
Creates a random individual by shuffling the order of objects in array
"""
def GenerateIndividual(creator, objects):
    return creator(random.sample(objects, len(objects)))

"""
Creates a 500 x 500 grid with objects organized inside containers.
Each container is a black square with white outline
Each object is a white rectangle with red outline
Max. 25 containers shown
Image is saved in /containers folder as container{id}.jpg
"""
def DrawContainers(containerSize, objects, id):
    # Resize for image visibility
    containerSize = tuple([10*x for x in containerSize])
    objects = tuple([(10*x[0], 10*x[1]) for x in objects])

    im = Image.new('RGB', (501, 501), (128, 128, 128))
    draw = ImageDraw.Draw(im)
    draw.rectangle((containerSize[0], containerSize[1], 0, 0), fill=(0, 0, 0), outline=(255, 255, 255))

    widthLeft = containerSize[0]
    heightLeft = containerSize[1]
    rowHeight = 0

    # Current container position
    containerX = 0
    containerY = 0

    # Current object position (0, 0) top left
    currX = 0
    currY = 0

    for obj in objects:
        # Adding object to current row
        if obj[0] <= widthLeft and obj[1] <= heightLeft:
            # Draw object in same row
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft -= obj[0]
            rowHeight = max(rowHeight, obj[1])
        # Adding new row
        elif obj[0] > widthLeft and obj[1] <= (heightLeft - rowHeight):
            # Draw object in new row
            currY += rowHeight
            currX = containerX
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft = containerSize[0] - obj[0]
            heightLeft -= rowHeight
            rowHeight = obj[1]
        # Adding new container
        else:
            # Draw new container
            containerX += containerSize[0]
            if containerX >= 500:
                containerX = 0
                containerY += containerSize[1]
            draw.rectangle((containerX, containerY, containerX+containerSize[0], containerY+containerSize[1]),
                            fill=(0, 0, 0), outline=(255, 255, 255))
            # Draw new object
            currY = containerY
            currX = containerX
            draw.rectangle((currX, currY, currX+obj[0], currY+obj[1]), fill=(255, 255, 255), outline=(255, 0, 0))
            currX += obj[0]

            widthLeft = containerSize[0] - obj[0]
            heightLeft = containerSize[1]
            rowHeight = obj[1]

    im.save('./containers/container' + str(id) + '.jpg', quality=95)

"""
# Display a dataframe
"""
def Display(df, title, color='r'):
    df.reset_index(drop=True, inplace=True)
    df_means = df.groupby(['gen']).agg({'min': {'mean', 'std'}})

    X = df['gen'].unique()
    means = df_means['min']['mean'].values
    deviations = df_means['min']['std'].values
    plt.plot(X, means, color=color)
    plt.plot(X, means - deviations, color=color, linestyle='dashed')
    plt.plot(X, means + deviations, color=color, linestyle='dashed')
    plt.xlabel("Generation")
    plt.ylabel("Space wasted")
    plt.title(title)

    plt.show()

### SOLUTION ###
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))  # space wasted and num. containers
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("individual", GenerateIndividual, creator.Individual, objects=objects)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selNSGA2)
toolbox.register("mate", MateObjs, creator=creator.Individual)
toolbox.register("mutate", MutSet)
toolbox.register("evaluate", Evaluate)

pop = toolbox.population(n=10)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0]) # 0 Space wated - 1 Num. containers
stats.register("min", np.min)
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("median", np.median)
stats.register("std", np.std)

### Main loop ###
N_ITER = 10
N_GEN = 500
MU = 50
LAMBDA = 100
CXPB = 0.7
MUTPB = 0.3
df = pd.DataFrame()

# MuPlusLambda
hof = tools.HallOfFame(1)
for i in range(N_ITER):
    res_MuPlusLambda, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, 
            N_GEN, stats=stats, verbose=False, halloffame=hof)
    
    df_aux = pd.DataFrame(log)
    df = pd.concat([df, df_aux])

# Results
currSolution = hof[0]
DrawContainers(containerSize, currSolution, 3)
spaceWasted, numContainers = FillContainers(containerSize, currSolution)
print("NO. OF CONTAINERS: ", numContainers)
print("SPACE WASTED: ", spaceWasted)
print("OPTIMAL ORDER: ", currSolution)

Display(df, "Space wasted in containers")

# Check for solution integrity, no objects have been lost/modified
currSolution = sorted(currSolution, key=lambda x: (x[0], x[1]))
objects = sorted(objects, key=lambda x: (x[0], x[1]))
print("INTEGRITY: ", currSolution == objects)