
import sys
import timeit
import random
import numpy as np
import matplotlib.pyplot as plt
'''
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
'''

def randbin1(d):
    return ''.join(str(random.randint(0, 1))
                for x in range(d))



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
    hammings = np.zeros((n,n))
    for i in range(n):

        string = V[i]
        for j in range(i,n):
            other_string = V[j]
            dist = hamming(string, other_string)
            hammings[i][j] = dist
            if dist <= k and i != j:
                E.append((dist, i+1, j+1))
                E.append((dist, j+1, i+1))

    V = list(range(1,n+1))
    #print(hammings)
    return V, E

'''
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

def test(n,k,m):

    strings = []

    for i in range(n):

        strings.append(randbin1(m))

    #V,E = graph(strings,n,k,m)
    #print(len(E))
    wrapped = wrapper(graph, (strings,n,k,m))
    t = timeit.Timer("graph", setup= "from __main__ import graph")


    return t.timeit()

    #sys.exit()


#parent = dict()
#rank = dict()
mean_of_times = []
nr_of_iterations = 100
number_of_nodes = [10,100,1000,10000]
k_values = [2]#[8,16,32, 50]
m = 7#50
for number in number_of_nodes:
    print("Running for number: %i" % number)
    for k in k_values:
        print("Running for k: %i" % k)
        times = []
        for n in range(nr_of_iterations):
            time = test(number,k,m)
            print(time)
            times.append(time)


        mean_of_times.append(np.mean(times))

print(mean_of_times)


nr_nodes = [10,50,100,500,1000,5000,10000]
k = 2
m = 7
iterations = 10

average_times = []
for n in nr_nodes:
    print("Running for n: %i" % n)
    strings = []
    for i in range(n):
        strings.append(randbin1(m))

    times = []
    for t in range(iterations):
        start = timeit.default_timer()
        graph(strings, n, k, m)
        stop = timeit.default_timer()
        time = stop-start
        times.append(time)
    av_t = np.mean(times)
    average_times.append(av_t)
    print(av_t)


print('Average time:', average_times)

x = np.asarray(nr_nodes)
y = np.asarray(average_times)

fit = np.poly1d(np.polyfit(x,y,2))
xf = np.linspace(0,10000,100)


plt.plot(x,y,'*b')
plt.plot(xf,fit(xf),'--')
plt.title('Time Complexity of String Matching Algorithm')
plt.xlabel('Numer of strings n')
plt.ylabel('Time (s)')
plt.show()
'''

n = 100
k = 2
nr_lengths = [10, 50, 100, 500, 1000, 1500, 2000]
iterations = 10



average_times = []
for m in nr_lengths:
    print("Running for m: %i" % m)
    strings = []
    for i in range(n):
        strings.append(randbin1(m))
        times = []
    for t in range(iterations):
        start = timeit.default_timer()
        graph(strings, n, k, m)
        stop = timeit.default_timer()
        time = stop-start
        times.append(time)
    av_t = np.mean(times)
    average_times.append(av_t)
    print(av_t)


print('Average time:', average_times)

x = np.asarray(nr_lengths)
y = np.asarray(average_times)

fit = np.poly1d(np.polyfit(x,y,1))
xf = np.linspace(0,2000,100)


plt.plot(x,y,'*b')
plt.plot(xf,fit(xf),'--')
plt.title('Time Complexity of String Matching Algorithm')
plt.xlabel('Length, m, of each string')
plt.ylabel('Time (s)')
plt.show()














#if __name__ == '__main__':

    #elements = []

    #parent = dict()
    #rank = dict()

    #for line in sys.stdin:
    #    elements.append(line)

    #strings = elements[1:-1]

    #length = len(elements)

    #n = int(elements[0])
    #k = int(elements[length-1])
    #m = int(len(strings[0]))

    #V,E = graph(strings,n,k,m)
    #wrapped = wrapper(kruskal, (V,E))
    #t = timeit.Timer("kruskal", setup="from __main__ import kruskal")
    #print(t.timeit())

    #min_tree = kruskal(V,E)
    #sorted = sorted(min_tree, key=lambda element: (element[1], element[2]))

    #for each in sorted:
    #    print(each[1], each[2])


    #sys.exit(0)