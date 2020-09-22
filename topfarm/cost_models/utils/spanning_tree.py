from numpy import argmin, array, sqrt
import sys
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
from numpy import newaxis as na


def mst(x, y):
    d_ij = np.hypot(x - x[:, na], y - y[:, na])
    Tcsr = minimum_spanning_tree(d_ij)
    sp = (Tcsr.toarray())
    return {tuple(i): sp[tuple(i)] for i in np.argwhere(sp)}


def spanning_tree(X, Y):
    sys.stderr.write("spanning_tree() is deprecated; use mst instead\n")

    """
    Calculate a minimum spanning tree distance for a layout.
    Minimum spanning tree heuristic algorithm from Topfarm0.

    Parameters
    ----------
    x, y: ndarray([n_wt])
            X,Y positions of the wind turbines

    :return: connections: dict of keys (i_wt, j_wt) and value the distance between the two wind turbine index

    the total cable length is simply sum(list(connections.values()))
    """
    n_wt = len(X)
    # X, Y = positions[:,0], positions[:,1]

    def dist(i, j):
        return sqrt((X[i] - X[j])**2.0 + (Y[i] - Y[j])**2.0)

    turblist = range(n_wt)
    connections = {}
    islands = []

    # Look for the smallest connections, and cluster them together.
    for i_wt in turblist:
        not_i_wt = list(filter(lambda x: x != i_wt, turblist))
        distances = sqrt((X[not_i_wt] - X[i_wt])**2.0 + (Y[not_i_wt] - Y[i_wt])**2.0)
        id = argmin(distances)
        closest_wt = not_i_wt[id]

        # Add the connection to the structure
        connections[(i_wt, closest_wt)] = distances[id]
        #   connections{closest_wt}(end+1) = i_wt

        # Add the turbine to an island
        found = False
        for iIS, island in enumerate(islands):
            # Check if it's nearest turbine is already in an island
            if closest_wt in island or i_wt in island:
                # Check if we have already found that the turbine pair is part of an island
                if found:
                    # Those two islands are connected (iIs & id_island)
                    # let's merge them
                    island += filter(lambda x: x != i_wt and x != closest_wt, islands[id_island])
                    del islands[id_island]

                found = True
                id_island = iIS
                # If the current wt is not in the island
                if i_wt not in island:
                    # Add it to the island
                    island.append(i_wt)

                # If the closest_wt is not in the island
                if closest_wt not in island:
                    # Add it to the island
                    island.append(closest_wt)

        # If no island connected to the turbine pair has been found,
        # then we create a new one
        if not found:
            # Create a new island
            islands.append([i_wt, closest_wt])

    # Connect the islands together

    while len(islands) > 1:
        # Look for the closest turbine that is not in the first island
        dist_list = array([[dist(i_wt, j_wt), i_wt, j_wt]
                           for i_wt in islands[0]
                           for j_wt in turblist
                           if j_wt not in islands[0]])
        amin = argmin(dist_list[:, 0])
        i_wt = int(dist_list[amin, 1])
        j_wt = int(dist_list[amin, 2])

        # Add the connection to the structure
        if (i_wt, j_wt) not in connections and (j_wt, i_wt) not in connections:
            connections[(i_wt, j_wt)] = dist_list[amin, 0]
        # if i_wt not in connections[j_wt]:
        #    connections[j_wt].append(i_wt)

        for i, island in enumerate(islands):
            if j_wt in island:
                islands[0] += island
                del islands[i]

    return connections
