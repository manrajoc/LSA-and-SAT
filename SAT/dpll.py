# imports

from random import choice, choices, sample, random, randint
import matplotlib.pyplot as plt
import sys, ctypes
from multiprocessing import Pool
from functools import reduce
from statistics import median
from collections import Counter


# function definitions

def binary_not(v):
    if v[0] == 'P':
        return '~' + v
    else:
        return v[1:]

def simplify(clauses, literal):
    to_be_removed = []

    for clause in clauses:
        if literal in clause:
            to_be_removed.append(clause)
        elif binary_not(literal) in clause:
            clau = list(clause)
            clau.remove(binary_not(literal))
            clau = tuple(clau)
            clauses.append(clau)
            to_be_removed.append(clause)

    clauses = [clause for clause in clauses if clause not in to_be_removed]
    return clauses


def dpll(clauses, r_calls = 0):
    r_calls += 1

    if len(clauses) == 0:
        return (True, r_calls)
    for clause in clauses:
        if len(clause) == 0:
            return (False, r_calls)

    for clause in clauses:
        if len(clause) == 1:
            return dpll(simplify(clauses, clause[0]), r_calls=r_calls)
        
    literals = set()
    for clause in clauses:
        for literal in clause:
            literals.add(literal)

    v = choice(list(literals))
    v_bar = binary_not(v)

    con, r_calls = dpll(simplify(clauses, v), r_calls=r_calls)
    if con:
        return (True, r_calls)
    else:
        return dpll(simplify(clauses, v_bar), r_calls=r_calls)


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

    var_list = ["P" + str(i) for i in range(n)]
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
    # workers = Pool(processes = 12)
    for m in range(int(n/4), 10*n+1, int(n/4)):
        total_solvable = 0
        total = j
        avg_flips = 0
        calls = []

        if failure < 2*j:
            for i in range(j):
                clauses = CNF(k, m, n)
                solvable, r_calls = dpll(clauses)
                total_solvable += solvable
                calls.append(r_calls)

        print("median calls:" ,median(calls))
        print(int((10*m) / n), "% Done")

        x.append(m/n)
        y1.append(total_solvable/total)
        y2.append(median(calls))

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
    ax[1].set_ylabel("Runtime(DPLL calls)")

    ax[0].plot(x, y1, label = "Probability", marker = 'o')
    ax[0].plot([4.3, 4.3], [0, 1], label = "X = 4.3", ls = '--', color = 'red')
    ax[1].plot(x, y2, label = "DPLL Calls", marker = 'o')

    ax[0].legend()
    ax[1].legend()
    fig.savefig("Out.png", dpi=200)