import math as m
import sys

train = open(sys.argv[1], "r+")
file2 = train.readlines()

final_data = []
for line in file2:
    final_data.append(line.split())

depth = int(sys.argv[3])

def getValues(data):
    values = []
    for i in range(len(data[0])):
        labels = []
        for line in data[1:]:
            if line[i] not in labels:
                labels.append(line[i])
        values.append(sorted(labels))
    return values

Values = getValues(final_data)

def entropy(data):
    dcount = 0
    rcount = 0
    head = 0
    for line in data:
        if line[-1].strip() == Values[-1][0]:
            dcount += 1
        elif line[-1].strip() == Values[-1][1]:
            rcount += 1
        else:
            head += 1
    total = rcount + dcount 
    if (total == 0):
        E = 0
    elif (rcount == 0):
        E = -((dcount/total)*m.log((dcount/total),2))
    elif (dcount == 0):
        E = -((rcount/total)*m.log((rcount/total),2))
    else:
        E = -((rcount/total)*m.log((rcount/total),2) + (dcount/total)*m.log((dcount/total),2))
    return E, rcount, dcount


def cond_entropy(data, i):
    E = entropy(data)[0]
    list1 = []
    list2 = []
    head = []
    for line in data:
        if (line[i] == Values[i][0]): #for == "n"
            list1.append(line)
        elif (line[i] == Values[i][1]): #for == "y"
            list2.append(line)
        else:
            head.append(line)
    e1 = entropy(list1)[0]
    e2 = entropy(list2)[0]
    if (len(list1)+len(list2) == 0):
        p1 = 0
        p2 = 0
    else:
        p1 = len(list1)/(len(list1)+len(list2))
        p2 = len(list2)/(len(list1)+len(list2))
    cond_entropy = E - (p1*e1+p2*e2)
    return cond_entropy

#implementing tree

class Node:
    def __init__(self, key, call, lcount, rcount):
        self.left = None
        self.right = None
        self.val = key
        self.call = call
        self.lcount = lcount
        self.rcount = rcount

def max_entropy(data):
    A = []
    for i in range(len(data[0]) -1):
        A.append(cond_entropy(data, i))
    return max(A), A.index(max(A))

def partition(data, var):
    list1 = [final_data[0]]
    list2 = [final_data[0]]
    head = []
    for line in data:
        if (line[data[0].index(var)] == Values[data[0].index(var)][0]):
            list1.append(line)
        elif (line[data[0].index(var)] == Values[data[0].index(var)][1]):
            list2.append(line)
        else:
            head.append(line)
    lcount = entropy(data)[1]
    rcount = entropy(data)[2]
    if lcount > rcount:
        call = Values[-1][1]
    else:
        call = Values[-1][0]  
    return list1, list2, call, lcount, rcount

if max_entropy(final_data)[0] > 0:
    varind = max_entropy(final_data)[1]
    X = partition(final_data, final_data[0][varind])
    root = Node(final_data[0][varind], X[2], X[3], X[4])


def build_tree(data, root, depth):
    part = partition(data, root.val)
    node1 = max_entropy(part[0]) #gives me 2 things - max entropy among remaining attributes based on partitioned data list1 of root node and index for that value
    node2 = max_entropy(part[1]) 
    M = partition(part[0], final_data[0][node1[1]]) #partitioned data, all values of left node based on list1
    N = partition(part[1], final_data[0][node2[1]]) #partitioned data, all values of right node based on list2
    if node1[0] > 0 and depth > 0:
        root.left = Node(final_data[0][node1[1]], M[2], M[3], M[4])
        build_tree(part[0], root.left, depth-1)
    if node2[0] > 0 and depth > 0:
        root.right = Node(final_data[0][node2[1]], N[2], N[3], N[4])
        build_tree(part[1], root.right, depth-1)

build_tree(final_data, root, depth)

test = open(sys.argv[2], "r+")
file3 = test.readlines()
final_test = []
for line in file3:
    final_test.append(line.split())


def Predict(line, root):
     if line[final_data[0].index(root.val)] == Values[final_data[0].index(root.val)][0]:
         if root.left == None:
             label = root.call
         else:
             return Predict(line, root.left)
     if line[final_data[0].index(root.val)] == Values[final_data[0].index(root.val)][1]:
         if root.right == None:
             label = root.call
         else:
             return Predict(line, root.right)
     return label


def printInorder(root):
    if root:
        printInorder(root.left)
        print(root.val, "\t",end="")
        printInorder(root.right)

Label_test = []
for line in final_test[1:]:
    Label_test.append(Predict(line, root))
    
file5 = open(sys.argv[5], "w")
for g in Label_test:
    file5.writelines(g + "\n")
file5.close()

Label_train = []
for line in final_data[1:]:
    Label_train.append(Predict(line, root))

file6 = open(sys.argv[4], "w")
for h in Label_train:
    file6.writelines(h + "\n")
file6.close()

def error(data, label):
    err = 0
    for i in range(len(data)-1):
        if data[i+1][-1] != label[i]:
            err += 1
    error = err/(len(data)-1)
    return error

file4 = open(sys.argv[6], "w")
file4.writelines("error(train): " + str(error(final_data, Label_train)) + "\n")
file4.writelines("error(test): " + str(error(final_test, Label_test)))
file4.close()
            
#file1 = open(sys.argv[1], "r+")
#file2 = file1.readlines()

