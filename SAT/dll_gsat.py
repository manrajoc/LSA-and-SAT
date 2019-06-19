# imports

from random import sample, choice
from sys import argv
from multiprocessing import Pool
from time import time
from functools import reduce
from matplotlib import pyplot as plt
from copy import deepcopy

# func defs

def cnf(k, m, n):
    unique_literals = [i for i in range(1, n+1)]
    clauses = []

    for i in range(m):
        s = sample(unique_literals, k)
        s = list(map(lambda l: choice([1, -1]) * l, s))
        clauses.append(s)

    return clauses


# GSAT --------------------------------------------------------------------------------------------

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


def gsat_auxi(c):
    return gsat(c, 7*n, 2*n, n)


#--------------------------------------------------------------------------------------------------
# DLL ---------------------------------------------------------------------------------------------

def dll(clauses):
    clauses = deepcopy(clauses)
    
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
        return dll(simplify(clauses, literal))

    vs = []
    for clause in clauses:
        for literal in clause:
            vs.append(literal)
    v = choice(vs)

    if dll(simplify(clauses, v)):
        return True
    else:
        return dll(simplify(clauses, -v))


def simplify(clauses, literal):
    result = deepcopy(clauses)
    r_c = []
    r_l = []

    for i, clause in enumerate(clauses):
        if literal in clause:
            r_c.append(i)
        elif -literal in clause:
            r_l.append(i)

    r_c = r_c[::-1]
    r_l = r_l[::-1]

    for i in r_l:
        result[i].remove(-literal)
    for i in r_c:
        del result[i]

    return result


# -------------------------------------------------------------------------------------------------

# driver

""" if __name__ == "__main__":

    c = [[1, 2, 3], [-1, 2, -3], [2], [-1, -2, -3]]

    print(dll(c)) """

if __name__ == "__main__":
    algo = argv[1]
    k, n, cases = [int(i) for i in argv[2:]]
    workers = Pool(processes = 6)
    x = []
    y1 = []
    y2 = []


    if algo == "gsat":
        algo_fn = gsat_auxi
    elif algo == "dll":
        algo_fn = dll
    else:
        quit(""" Incorrect option for algorithm! \nPlease use either "gsat" or "dll" """)
    

    for m in range(int(n/4), 8*n+1, int(n/4)):

        inputs = [cnf(k, m, n) for i in range(cases)]

        start_time = time()
        output = workers.map(algo_fn, inputs)
        call_time = time() - start_time

        solvable = reduce(lambda x, y: x+y, output)

        y1.append(solvable / cases)
        y2.append(call_time)
        x.append(m/n)

        print(m/n)

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
    ax[1].set_ylabel("Runtime(s)")

    ax[0].plot(x, y1, label = "Probability", marker = 'o')
    ax[0].plot([4.3, 4.3], [0, 1], label = "X = 4.3", ls = '--', color = 'red')
    ax[1].plot(x, y2, label = "Runtime in seconds", marker = 'o')

    ax[0].legend()
    ax[1].legend()
    fig.savefig("Out.png", dpi=200)