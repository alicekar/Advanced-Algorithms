import numpy as np
import sys
import fileinput



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
	hammings = np.zeros((n,n))+np.inf
	E = []
	correct_parents = []
	# List containing the corresponding indicies of E 
	upper_graph = []
	lower_graph = []
	for i in range(n):
		string = V[i]
		for j in range(i,n):
			other_string = V[j]
			dist = hamming(string,other_string)
			if dist <= k and i != j:
				hammings[i][j] = dist
				#hammings[j][i] = dist
				E.append((string, other_string))
				w = dist
				upper_graph.append(((i,j),w))
				correct_parents.append(i)
				correct_parents.append(j)
				lower_graph.append(((j,i),w))

	graph = [upper_graph, lower_graph]
	correct_parents = np.unique(np.asarray(correct_parents))
	correct_parents = correct_parents.tolist()
	G = (V,E)
	print(hammings)
	print(graph)
	prims(graph, correct_parents)



def find_children(parents, graph):
	upper_children = []
	lower_children = []
	upper_graph, lower_graph = graph
	for parent in parents:
		for node in upper_graph:
			if node[0][0] == parent and node[0][1] not in parents:
				upper_children.append(node)
		for node in lower_graph:
			if node[0][0] == parent and node[0][1] not in parents:
				lower_children.append(node)

	children = [upper_children, lower_children]
	return(children)



def find_optimal(children):
	upper_children, lower_children = children
	l1 = len(upper_children)
	l2 = len(lower_children)
	upper_children = np.asarray(upper_children)
	lower_children = np.asarray(lower_children)
	candidates = []
	#print(children)
	if l1 != 0:
		weights = upper_children[:,1]
		indicies = upper_children[:,0]
		indicies = np.asarray([ind for sublist in indicies for ind in sublist]).reshape((l1,2))
		if len(set(weights))==1:
			if len(set(indicies[:,0]))==1:
				best1 = upper_children[np.argmin(indicies[:,1])]
				candidates.append(best1.tolist())
			else:
				best1 = upper_children[np.argmin(indicies[:,0])]
				candidates.append(best1.tolist())
		else: 
			best1 = upper_children[np.argmin(upper_children[:,1])]
			candidates.append(best1.tolist())
		#return(best1[0])
	else:
		best1 = None
		candidates.append(best1)
		#return(None)

	print(lower_children)
	if l2 != 0:
		weights = lower_children[:,1]
		indicies = lower_children[:,0]
		indicies = np.asarray([ind for sublist in indicies for ind in sublist]).reshape((l2,2))
		if len(set(weights))==1:
			if len(set(indicies[:,1]))==1:
				best2 =lower_children[np.argmin(indicies[:,0])]
				candidates.append(best2.tolist())
			else:
				best2 = lower_children[np.argmin(indicies[:,1])]
				candidates.append(best2.tolist())
		else: 
			best2 = lower_children[np.argmin(lower_children[:,1])]
			candidates.append(best2.tolist())
		#return(best2[0])
		
	else:
		best2 = None
		candidates.append(best2)
		#return(None)

	#candidates = [None, None]

	if candidates[0] != None and candidates[1] != None:
		#M = np.asarray([ind for sublist in candidates for ind in sublist]).reshape((2,2))
		b1 = candidates[0]
		b2 = candidates[1]
		if b1[1] == b2[1]:
			if b1[0][0]==b2[0][1]:
				if b1[0][1] <= b2[0][0]:
					best = b1
					return(best[0])
				else:
					best = b2
					return(best[0])
			elif b1[0][0] < b2[0][1]:
				best = b1
				return(best[0])
			else:
				best = b2
				return(best[0])
		elif b1[1]<b2[1]:
			best = b1
			return(best[0])
		else:
			best = b2
			return(best[0])
	elif candidates[0] != None and candidates[1] == None:
		best = candidates[0]
		return(best[0])
	elif candidates[0] == None and candidates[1] != None:
		best = candidates[1]
		return(best[0])
	else:
		best = None
		return(best)


	#if len(best1[0]) == len(best2[0]):
	#	if best1[1]==best2[1]:
	#		if best1[0][0] == best2[0][1]:

	#print(best1[0])
	#print(best2[0])
	#if best1 != 2:
		#print('ja')
	#return(best1[0],best2[0])


def prims(graph, correct_parents):
	controll = correct_parents
	print(controll)
	#print(graph)
	visited = []
	result = []
	if len(graph[0]) != 0 or len(graph[1]) != 0:
		parent = graph[0][0][0][0]	
		visited.append(parent)
		'''
		children = find_children(visited, graph)
		print(children)
		op = find_optimal(children)
		#print(op)
		'''

	
		while len(visited) != len(controll):
			children = find_children(visited, graph)
			if len(children[0]) == 0 and len(children[1])==0:
				others = list(set(controll)-set(visited))
				separate_parents = []
				for node in others:
					parent = node
					separate_parents.append(parent)
					visited.append(parent)
					children = find_children(separate_parents, graph)
					op = find_optimal(children)
					if op == None:
						break
					else:
						result.append(op)
						#print(node)
			else:
				optimal = find_optimal(children)
				result.append(optimal)
				parent = optimal[1]
				visited.append(parent)
				#print(visited)
				#print(optimal)


	
		result = np.asarray(result)+1
		#print(np.sort(np.asarray(visited)).tolist())
		#print(len(result))
		#print(np.array_equal(np.asarray(controll),np.sort(np.asarray(visited))))
		#final = final[final[:,1].argsort()]
		final = np.sort(result, axis=1)
		final = np.unique(final, axis=0)
		print(len(final))
		for f in final:
			print(int(f[0]),int(f[1]))
	else:
		print()
	
	


if __name__ == '__main__':
	elements = []
	for line in sys.stdin:
		elements.append(line.strip('\n'))

	strings = elements[1:-1]
	length = len(elements)
	n = int(elements[0])
	k = int(elements[length-1])
	m = int(len(strings[0]))

	graph(strings,n,k,m)


