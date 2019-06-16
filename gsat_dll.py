# imports

from random import sample, choice
from pysat.solvers import Glucose3
from sys import argv
from multiprocessing import Pool
from time import time
from functools import reduce
from matplotlib import pyplot as plt

# func defs

def cnf(k, m, n):
    unique_literals = [i for i in range(1, n+1)]
    clauses = []

    for i in range(m):
        s = sample(unique_literals, k)
        s = list(map(lambda l: choice([1, -1]) * l, s))
        clauses.append(s)

    return clauses

def gsat(clauses, max_flips, max_restarts, n):
    
    for i in range(max_restarts):

        # create a random model
        model = [choice([True, False]) for i in range(1, n+1)]

        if resolve(clauses, model):
            return True

        for j in range(max_flips):
            model = optimal_flip(clauses, model)

            if resolve(clauses, model):
                return True

    return False


def resolve(clauses, model):
    
    r_cnf = True
    for clause in clauses:
        r_clause = False
        
        for literal in clause:
            if literal > 0:
                if model[abs(literal) - 1]:
                    r_clause = True
            else:
                if not model[abs(literal) - 1]:
                    r_clause = True

        if not r_clause:
            r_cnf = False

    return r_cnf

def optimal_flip(clauses, model):

    flip_scores = []

    for i in range(len(model)):
        model[i] = not model[i]
        score = 0

        for clause in clauses:
            if resolve([clause], model):
                score += 1

        flip_scores.append(score)
        model[i] = not model[i]

    index = flip_scores.index(max(flip_scores))
    model[index] = not model[index]

    return model

# DLL ---------------------------------------------------------------------

def dll(clauses):
    
    if len(clauses) == 0:
        return True
    
    if len([True for clause in clauses if len(clause) == 0]) > 0:
        return False

    unit_clauses = []
    for clause in clauses:
        if len(clause) == 1:
            unit_clauses.append(clause)

    if len(unit_clauses) > 0:
        literal = choice(unit_clauses)[0]
        pass


def simplify(clauses, literal):

    for clause in clauses:
        if literal in clause:
            clauses.remove(clause)

        if -literal in clause:
            clause.remove(-literal)

    return clauses

# -------------------------------------------------------------------------

# driver
def gsat_auxi(c):
    return gsat(c, 7*n, 2*n, n)


if __name__ == "__main__":
    cases, n = [int(v) for v in argv[1:]]
    workers = Pool(processes = 12)
    x = []
    y1 = []
    y2 = []

    for m in range(int(n/2), int(6*n), int(n/2)):

        input = [cnf(3, m, n) for i in range(cases)]

        start = time()
        output = workers.map(gsat_auxi, input)
        y2.append(time() - start)

        y1.append(reduce(lambda x, y: x + y, output)/cases)
        x.append(m/n)
        
        print((m/n))

    

# Plotting via matplotlib
    fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
    ax[0].grid(b=True, which='major', color='#666666', linestyle='-')
    ax[0].minorticks_on()
    ax[0].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax[1].grid(b=True, which='major', color='#666666', linestyle='-')
    ax[1].minorticks_on()
    ax[1].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    ax[0].set_title("Probability vs Clause/Symbol")
    ax[0].set_xlabel("Clause/Symbol Ratio or m/n")
    ax[0].set_ylabel("Probability")
    ax[1].set_title("Runtime")
    ax[1].set_xlabel("Clause/Symbol Ratio or m/n")
    ax[1].set_ylabel("runtime in microseconds")

    ax[0].plot(x, y1, label = "Probability", marker = 'o')
    ax[0].plot([4.3, 4.3], [0, 1], label = "X = 4.3", ls = '--', color = 'red')
    ax[1].plot(x, y2, label = "Runtime in μs", marker = 'o')

    ax[0].legend()
    ax[1].legend()
    fig.savefig("Out.png", dpi=200)