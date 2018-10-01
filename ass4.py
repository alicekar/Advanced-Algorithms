import sys
import numpy as np


#suffix array for the strings
#k+1 pieces of the array
#FInd pices that matches, slit check Hash tabel or suffic arrays.
#More complex, LCP array Longest common prefix. LCP[i] > n/(k+1)
def find_distance(string1, string2, m):
	matrix = np.zeros((m+1,m+1))
	matrix[0] = np.arange(int(string2))
	#matrix[:,0] = np.arange(m+1)
	print(string2)
	print(string1)

	for i in range(1,m+1): #string1 - along colum
		for j in range(1,m+1):
			if string1[i] == string2[j]:
				matrix[i][j] = matrix[i-1][j-1]
			else:
				matrix[i][j] = matrix[i-1][j-1]+1

	print(matrix)



def hamming(string1, string2):
	tuple_list = list(zip(string1,string2))
	if len(string1) != len(string2):
		return('lenths do not match')
	else: 
		dist = 0
		for n1, n2 in tuple_list:
			if n1 != n2:
				dist += 1
	return(dist)


def graph(S, n, k, m):
    V = S
    E = []
    hammings = np.zeros((n,n))+np.inf
    for i in range(n):
        string = V[i]
        for j in range(i,n):
            other_string = V[j]
            dist = hamming(string, other_string)

            if dist <= k and i != j:
                E.append((i+1, j+1, dist))
                hammings[i][j] = dist
                #E.append((dist, j+1, i+1))

    V = list(range(1,n+1))
    #return V, E




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
        m = int(len(strings[0]))-1
       
        #graph(strings,n,k,m)
        find_distance(strings[0],strings[1],m)




