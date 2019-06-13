# imports

from random import choice, sample, random
import matplotlib.pyplot as plt
import sys
from time import time
from statistics import median
from pysat.solvers import Glucose3


# function definitions

def sat_solver(clauses):
    """ Wrapper function for Glucose3 SAT solver, for custom style clauses input
        
        Input:  list of clauses in the form returned by CNF function
        Output: boolean value representing satisfiability and runtime in μs """

    g = Glucose3()

    for clause in clauses:
        clau = list()

        for literal in clause:
            if literal[0] == 'P':
                clau.append(int(literal[1:]))
            else:
                clau.append(-int(literal[2:]))

        g.add_clause(clau)

    start_time = time()
    s = g.solve()
    call_time = (time() - start_time) * 10**6
    return s, call_time

def CNF(k, m, n):
    """ Creates a random Conjunctive Normal Form expression
    
    Input:  k(integer), m(integer), n(integer)
    
            k:  the number of literals per clause
            m:  number of clauses in the sentence
            n:  the number of unique literals in the sentence
            
    Example Input:  k = 3
                    m = 50
                    n = 50
                    
    Output: Returns a list of clauses where each clause is a tuple of literals """

    var_list = ["P" + str(i) for i in range(1, n+1)]
    not_list = ["~", ""]
    clause_set = set()

    while len(clause_set) != m:
        s = sample(var_list, k)
        for i in range(len(s)):
            s[i] = choice(not_list) + s[i]

        clause = tuple(s)
        clause_set.add(clause)

    return list(clause_set)


# Driver Code
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect syntax, Quitting\n")
        print("Correct Syntax: python3 main.py literal_per_clause unique_literals iterations_for_probability")
        quit()
    else:
        k, n, j = [int(i) for i in sys.argv[1:]]

    x = []
    y1 = []
    y2 = []

    failure = 0
    for m in range(int(n/4), 10*n+1, int(n/4)):
        total_solvable = 0
        total = j
        avg_flips = 0
        calls = []

        if failure < 2*j:
            for i in range(j):
                clauses = CNF(k, m, n)
                solvable, call_time = sat_solver(clauses)
                calls.append(call_time)
                total_solvable += solvable

        print(median(calls))
        print(int((10*m) / n), "% Done")

        x.append(m/n)
        y1.append(total_solvable/total)
        y2.append(median(calls))

    # PLOT STUFF
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