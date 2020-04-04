import argparse
import csv
from collections import Counter
import math

def readfile(file_path):
    trials = []
    count = []
    with open(file_path, 'r') as file:
        read = csv.reader(file, delimiter=',')
        for row in read:
            trials.append(row)
            row = list(row[0])
            result_count = Counter(row)
            count.append([result_count['0'], result_count['1']])
    
    return trials, count

def OnlineLearning(trials, count, A, B):
    for i in range(len(trials)):
        print('case {}: '.format(i + 1), trials[i][0])
        N = count[i][0] + count[i][1] # the number of the trials in this case
        m = count[i][1] # the number of head
        MLE = m / N
        likelihood = (math.factorial(N) / (math.factorial(m) * math.factorial(N-m))) * (MLE ** m) * ((1 - MLE) ** (N-m))
        print('Likelihood: ', likelihood)
        print('Beta prior:     a = {} b = {}'.format(A, B))
        A += count[i][1]
        B += count[i][0]
        print('Beta posterior: a = {} b = {}'.format(A, B), '\n')

# default
file_path = 'testfile.txt'
A = 0
B = 0

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--TESTFILE", type=str)
parser.add_argument("--A", type=int)
parser.add_argument("--B", type=int)
args = parser.parse_args()
file_path = args.TESTFILE
A = args.A
B = args.B

# read file
trials, count = readfile(file_path)
# online learning
OnlineLearning(trials, count, A, B)

