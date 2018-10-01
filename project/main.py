
import sys
import timeit

def hamming(string1, string2):
    tuple_list = list(zip(string1,string2))
    if len(string1) != len(string2):
        return None
    else:
        dist = 0
        for n1, n2 in tuple_list:
            if n1 != n2:
                dist += 1
    return(dist)

#suffix array for the strings
#k+1 pieces of the array
#FInd pices that matches, slit check Hash tabel or suffic arrays.
#More comples, LCP array Longest commen prefix. LCP[i] > n/(k+1)
def graph(S, n, k, m):
    V = S
    E = []
    for i in range(n):
        string = V[i]
        for j in range(i,n):
            other_string = V[j]
            dist = hamming(string, other_string)

            if dist <= k and i != j:
                E.append((dist, i+1, j+1))
                E.append((dist, j+1, i+1))

    V = list(range(1,n+1))

    return V, E


def find(vertice):
    if parent[vertice] != vertice:
        parent[vertice] = find(parent[vertice])
    return parent[vertice]

def union(vertice1, vertice2):
    root1 = find(vertice1)
    root2 = find(vertice2)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
        if rank[root1] == rank[root2]:
            rank[root2] += 1


def kruskal(V,E):

    for vertice in V:
        parent[vertice] = vertice
        rank[vertice] = 0

    minimum_spanning_tree = set()
    E.sort()

    for edge in E:
        weight, vertice1, vertice2 = edge

        # If the current edge connects two unconnected partitions
            # add the edge to the minimum spanning tree
            # megre the two partitions
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)

    return minimum_spanning_tree

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

if __name__ == '__main__':

    elements = []

    parent = dict()
    rank = dict()

    for line in sys.stdin:
        elements.append(line)

    strings = elements[1:-1]

    length = len(elements)

    if length != 0:
        n = int(elements[0])
        k = int(elements[length-1])
        m = int(len(strings[0]))

        V,E = graph(strings,n,k,m)
        #wrapped = wrapper(kruskal, (V,E))
        #t = timeit.Timer("kruskal", setup="from __main__ import kruskal")
        #print(t.timeit())

        min_tree = kruskal(V,E)
        sorted = sorted(min_tree, key=lambda element: (element[1], element[2]))

        for each in sorted:
            print(each[1], each[2])


        sys.exit(0)


        