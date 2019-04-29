import numpy as np

def average_precision(scores, golds, limit = None):
	pairs = list(zip(scores, golds))
	pairs = sorted(pairs, key=lambda x: x[0])
	pairs.reverse()

	count_pos = 0.0
	precisions = []
	for i in range(len(pairs)):
		if limit is not None and i == limit:
			break
		if pairs[i][1] == 1:
			count_pos += 1.0
			precisions.append(count_pos / float(i+1))
	
	return sum(precisions) / float(len(precisions))