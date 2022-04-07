#!/usr/bin/python3
import copy

from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
    from PyQt6.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        rcm = np.full((ncities, ncities), float('inf'), dtype=float)  # todo faster if int
        # Fill cost matrix
        for i in range(ncities):
            for j in range(ncities):
                rcm[i][j] = cities[i].costTo(cities[j])  # TODO convert to int?

        while not foundTour and time.time() - start_time < time_allowance:
            newRCM = rcm.copy()
            curCity = random.randint(0, ncities - 1)  # get random starting city
            newRCM[:, curCity] = float(
                'inf')  # prevent any paths from returning here before all cities have been reached
            route = [curCity]
            for i in range(ncities):  # for each city
                nextCity = int(np.where(newRCM[curCity] == np.amin(newRCM[curCity]))[0][
                                   0])  # get index of minpath out of this city
                newRCM[:, nextCity] = float('inf')  # prevent any other incoming edges to this city
                route.append(nextCity)  # add to route
                curCity = nextCity  # update current city
            bssf = TSPSolution([cities[x] for x in route])  # convert to tsp solution
            count += 1
            if bssf.cost < np.inf:  # check if solution is valid
                # Found a valid route
                foundTour = True

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        # Initialize the variables
        debug = False
        count = 0
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        maxHeapSize = 0
        stateCount = 0
        numPruned = 0
        minHeap = []
        bssf = self.greedy(time_allowance)['soln']
        start_time = time.time()

        # Create the initial state
        rootMatrix = [[float('inf') for x in range(ncities)] for y in range(ncities)]
        for i in range(ncities):

            for j in range(ncities):
                rootMatrix[i][j] = cities[i].costTo(cities[j])

        # Reduce state and add to heap
        matrix, reducedCost = self.reduceMatrix(rootMatrix)
        heapq.heappush(minHeap, (reducedCost, matrix, [0]))

        # While there are states in the heap
        while (debug or time.time() - start_time < time_allowance) and len(minHeap) > 0:

            cost, matrix, currPath = heapq.heappop(minHeap)

            # If the cost higher than bssf or the depth is less than halfway, then the state is pruned
            if cost < bssf.cost or len(currPath) > ncities / 2:
                for j in range(ncities):

                    if time.time() - start_time >= time_allowance:
                        break

                    # If this potential path is greater than the current bssf, then we ignore it.
                    if matrix[currPath[-1]][j] < np.inf and matrix[currPath[-1]][j] < bssf.cost - cost:
                        jumpCost = matrix[currPath[-1]][j]
                        newMatrix = self.createNewMatrix(copy.deepcopy(matrix), currPath[-1], j)
                        newMatrix, reductionCost = self.reduceMatrix(newMatrix)
                        newPath = copy.deepcopy(currPath)
                        newPath.append(j)
                        totalCost = cost + jumpCost + reductionCost

                        # If this new path is a complete path and is better than the current bssf, then we update bssf
                        if len(newPath) == ncities:
                            count = count + 1
                            if bssf.cost > totalCost:
                                pathCities = []
                                for x in range(len(newPath)):
                                    pathCities.append(cities[newPath[x]])
                                bssf = TSPSolution(pathCities)
                                foundTour = True
                                continue

                        # If this new path is not a complete path, then we add it to the heap if cost < bssf
                        if bssf.cost > totalCost:
                            heapq.heappush(minHeap, (totalCost, newMatrix, newPath))
                            stateCount = stateCount + 1
                            if len(minHeap) > maxHeapSize:
                                maxHeapSize = len(minHeap)
            else:
                numPruned += 1

        # If we found a tour, then we return the results else, return the random bssf we found earlier
        end_time = time.time()
        if end_time >= 60:
            for i in range(len(minHeap)):
                if minHeap[i][0] > bssf.cost:
                    numPruned += 1

        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = maxHeapSize
        results['total'] = stateCount
        results['pruned'] = numPruned
        return results

    def reduceMatrix(self, matrix):
        cost = 0
        # reduce row wise
        for i in range(len(matrix)):
            rowMin = min(matrix[i])
            if rowMin == float('inf') or rowMin == 0:
                continue
            for j in range(len(matrix[i])):
                matrix[i][j] -= rowMin
            cost += rowMin

        # reduce column wise
        for j in range(len(matrix)):
            temp = []
            for i in range(len(matrix)):
                temp.append(matrix[i][j])
            minCol = min(temp)
            # The or infinity could be a problem.
            if minCol == 0 or minCol == float('inf'):
                continue
            for i in range(len(matrix)):
                matrix[i][j] -= minCol
            cost += minCol

        return matrix, cost

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''
    def fancy(self, time_allowance=60.0, NUM_ANTS=20, PHEROMONE_WEIGHT=1.0, STANDARD_WEIGHT=2.0, WEIGHT_CONSTANT=0.01,
              CHECK_FOR_CONVERGENCE=250, distance_adjustment=10, PERSONALITY_WEIGHT=1.0):
        # Functions
        homesickness = 2
        def get_transition_probability(idx1, idx2, antType, depth, distanceMatrix, minDistance, maxDistance):
            if antType == 'homesick':
                if depth < ncities/homesickness:
                    return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT) * pow(distanceMatrix[idx1][idx2]/maxDistance, -PERSONALITY_WEIGHT)
                else:
                    return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT) * pow(distanceMatrix[idx1][idx2]/maxDistance, PERSONALITY_WEIGHT)
            elif antType == 'lonely':
                return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT)
            elif antType == 'extrovert':
                return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT)
            else:
                return pow(edge_weights[idx1][idx2], PHEROMONE_WEIGHT) * pow(edge_distances[idx1][idx2], -STANDARD_WEIGHT)

        def get_probablistic_path_from(source, antType):
            path = []
            dist = 0.0
            path.append(source)
            curr_idx = source

            distanceMatrix = []
            xSortCities = []
            ySortCities = []

            minDistance = float('inf')
            maxDistance = float('-inf')
            if antType == 'homesick':
                distanceMatrix = [[float('inf') for x in range(ncities)] for y in range(ncities)]
                for i in range(ncities):
                    for j in range(ncities):
                        distanceMatrix[i][j] = cities[i].distanceTo(cities[j])
                        if distanceMatrix[i][j] < minDistance:
                            minDistance = distanceMatrix[i][j]
                        if distanceMatrix[i][j] > maxDistance:
                            maxDistance = distanceMatrix[i][j]

            if antType == 'extrovert':
                xSortCities = sorted(cities, key=lambda city: city._x)
                ySortCities = sorted(cities, key=lambda city: city._y)



            while len(path) < ncities:
                n_sum = 0.0
                possible_next = []
                for n in range(ncities):
                    if n in path or edge_distances[curr_idx][n] == float('inf'):  # already visited or inf
                        continue
                    n_sum += get_transition_probability(curr_idx, n, antType, len(path), distanceMatrix, minDistance, maxDistance)
                    possible_next.append(n)

                if len(possible_next) == 0:  # avoid getting caught when no more possible edges
                    return path, float('inf')

                r = np.random.uniform(0.0, n_sum)
                x = 0.0
                for nn in possible_next:
                    x += get_transition_probability(curr_idx, nn, antType, len(path), distanceMatrix, minDistance, maxDistance)
                    if r <= x:
                        dist += edge_distances[curr_idx][nn]
                        curr_idx = nn
                        path.append(nn)
                        break
            dist += edge_distances[curr_idx][source]
            return path, dist

        # Initialize variables
        # NUM_ANTS = 100
        # PHEROMONE_WEIGHT = 1.0  # pheromone weight
        # STANDARD_WEIGHT = 2.0  # greedy weight
        # WEIGHT_CONSTANT = 0.01
        # CHECK_FOR_CONVERGENCE = 100 # Run this many loops before checking if no progress made
        cities = self._scenario.getCities()
        ncities = len(cities)
        converged = False
        count = 0
        old_bssf = 100000
        new_bssf = 100000
        bssf_path = []
        start_time = time.time()
        edge_distances = np.full((ncities, ncities), float('inf'), dtype=float)
        # Fill weight matrix
        for i in range(ncities):
            for j in range(ncities):
                edge_distances[i][j] = cities[i].costTo(cities[j])
        edge_distances += distance_adjustment  # Todo this is so that the weight isn't inf
        edge_weights = np.full((ncities, ncities), 1.0, dtype=float)

        # Main loop
        while not converged and time.time() - start_time < time_allowance:
            for k in range(CHECK_FOR_CONVERGENCE):
                # Evaporation step
                edge_weights *= 0.999
                new_weights = edge_weights.copy()
                # run all ants
                for j in range(NUM_ANTS):

                    cost = float('inf')  # Get a valid path
                    while cost == float('inf'):
                        homeCity = random.randint(0, ncities - 1)
                        antpath, cost = get_probablistic_path_from(homeCity, "homesick")

                    if cost < new_bssf:
                        new_bssf = cost
                        bssf_path = antpath

                    diff = cost - new_bssf + 0.05  # avoid a difference of 0 for division by 0 error todo use new_bssf?
                    weight_update = WEIGHT_CONSTANT / diff

                    for i in range(ncities):  # todo needs + 1?
                        source = antpath[i % ncities]
                        dest = antpath[(i + 1) % ncities]
                        new_weights[source][dest] += weight_update  # update weight for edge in both directions
                        new_weights[dest][source] += weight_update
                # update edge weights and normalize to sum to 1
                for i in range(ncities):
                    # rowsum = 0.0
                    # for j in range(ncities):
                    #     if i == j:
                    #         continue
                    #     rowsum += new_weights[i][j]
                    rowsum = sum(new_weights[i])  # todo does this work?
                    for j in range(ncities):
                        # multiplying by 2 since every node has two neighbors eventually
                        edge_weights[i][j] = 2 * new_weights[i][j] / rowsum
            # check for converge
            if new_bssf == old_bssf:
                converged = True
            old_bssf = new_bssf

        solution = TSPSolution([cities[x] for x in bssf_path])  # create the solution
        # if solution.cost != new_bssf:
        #     exit(-2) # This should never happen
        # Return values
        results = {}
        end_time = time.time()
        results['cost'] = solution.cost
        results['time'] = end_time - start_time
        results['count'] = None  # updatesToBSSF
        results['soln'] = solution
        results['max'] = None  # maxQueueSize
        results['total'] = None  # numPruned + numStatesCreated
        results['pruned'] = None  # numPruned
        return results

    def createNewMatrix(self, matrix, i, j):
        for x in range(len(matrix)):
            matrix[x][j] = float('inf')
            matrix[i][x] = float('inf')
        return matrix
