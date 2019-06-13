# imports

from random import choice, choices, sample, random, randint
import matplotlib.pyplot as plt
import sys, ctypes
from multiprocessing import Pool
from functools import reduce


# function definitions

def WALKSAT(clauses, p, max_flips, n):
    """ Implements the WALKSAT algorithm
    
    Input:  clauses(list), p(float), max_flips(integer), n(integer)
    
            clauses:    list of clauses, with clauses as tuples of literals
            p:          float value between 0-1 representing the probability of 
                        making a random flip instead of an optimal one, consequently
                        1-p represnts probability of optimal flip
            max_flips:  integer value representing the maximum number of truth values
                        the algorithm can flip before giving up
            n:          integer value representing the number of unique literals
                        present in the CNF/clauses
                        
                        
    Example Input:  clauses = [("P1", "P0"), ("P2", "~P0"), ("P2", "P1")]
                    p = 0.5
                    max_flips = 50
                    n = 3
                    
    Output: Returns a boolean value representing the existence of a solution
            and the number of value flips done before finding the solution """
    
    model = [choice([True, False]) for i in range(n)]
    flips = 0

    for i in range(max_flips):
        if resolve_cnf(clauses, model):
            return True, flips

        clause = choice(false_clauses(clauses, model))

        if choices([True, False], weights=[p, 1-p]):
            literal_index = randint(0, len(clause) - 1)
            literal_model = int(clause[literal_index][-1])
            model[literal_model] = not model[literal_model]
        else:
            model = optimal_flip(clause, model)

        flips = i+1

    return (False, flips)


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


def optimal_flip(clause, model):
    """ Flips the variable which maximizes the number of satisfied clauses

        Input:  clauses(list), model(list)
        
                clauses:    list of clauses with clauses as tuples of literals
                model:      list of truth values for each literal mapped by index values,
                            i.e. value at index 0 is the value of literal P0

        Output: returns model with flipped variable """
    
    unsatisfied_clauses = []

    for literal in clause:
        model_temp = model
        model_temp[int(literal[-1])] = not model_temp[int(literal[-1])]
        unsatisfied_clauses.append(false_clauses(clauses, model_temp, count=True))

    optimal_var = unsatisfied_clauses.index(min(unsatisfied_clauses))
    model[optimal_var] = not model[optimal_var]
    
    return model

def resolve_cnf(clauses, model):
    """ Resolves a Conjunctive Normal Form expression's truth value
    
        Input:  clauses(list), model(list)
        
                clauses:    list of clauses with clauses as tuples of literals
                model:      list of truth values for each literal mapped by index values,
                            i.e. value at index 0 is the value of literal P0
                            
        Example Input:  clauses = [("P1", "P0"), ("P2", "~P0"), ("P2", "P1")]
                        model = [True, False, False] 
                        
        Output: Returns a single boolean value, a True or a False """

    eval_str = ""

    for clause in clauses:
        eval_str += "("

        for literal in clause:
            eval_str += "("

            if literal[0] == "~":
                eval_str += str(not model[int(literal[2])])
            else:
                eval_str += str(model[int(literal[1])])
            
            eval_str += ")or"

        eval_str = eval_str[:-2]
        eval_str += ") and "

    eval_str = eval_str[:-5]

    return eval(eval_str)


def false_clauses(clauses, model, count=False):
    """ Looks for unsatisfied clauses, i.e. clauses that have truth value as false
        and returns the unsatisfied clauses or number of unstatisfied clauses depending on "mode" 
        
        Input:  clauses(list), model(list), mode(integer)
        
                clauses:    list of clauses with clauses as tuples of literals
                model:      list of truth values for each literal mapped by index values,
                            i.e. value at index 0 is the value of literal P0
                count:      boolean value, if true, the function returns count of unstatisfied clauses
                            and if false returns the list of unstatisfied clauses
                            
        Example Input:  clauses = [("P1", "P0"), ("P2", "~P0"), ("P2", "P1")]
                        model = [True, False, False] 
                        
        Output: Returns a list of unsatisfied clauses or their count """
    
    false_list = []

    for clause in clauses:
        if not resolve_cnf([clause], model):
            false_list.append(clause)

    if not count:
        return false_list
    else:
        return len(false_list)

# Misc

def auxi_walksat(clauses):
    return WALKSAT(clauses, p, max_flips, n)


# Driver Code
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Incorrect syntax, Quitting\n")
        print("Correct Syntax: python3 main.py literal_per_clause unique_literals max_flips iterations_for_probability probability_of_random_selection")
        quit()
    else:
        k, n, max_flips, j = [int(i) for i in sys.argv[1:-1]]
        p = float(sys.argv[-1])

    x = []
    y1 = []
    y2 = []

    failure = 0
    workers = Pool(processes = 12)
    for m in range(int(n/4), 10*n+1, int(n/4)):
        total_solvable = 0
        total = j
        avg_flips = 0

        if failure < 2*j:
            inputs = [CNF(k, m, n) for i in range(j)]
            outputs = workers.map(auxi_walksat, inputs)
            outputs = list(outputs)

            solvable, flips = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), outputs)
            total_solvable += solvable
            avg_flips += flips

            if not solvable:
                failure += j
            else:
                failure = 0


        print(int((10*m) / n), "% Done")

        x.append(m/n)
        y1.append(total_solvable/total)
        y2.append(avg_flips/j)

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
    ax[1].set_ylabel("Runtime(flips)")

    ax[0].plot(x, y1, label = "Probability", marker = 'o')
    ax[0].plot([4.3, 4.3], [0, 1], label = "X = 4.3", ls = '--', color = 'red')
    ax[1].plot(x, y2, label = "Runtime in flips", marker = 'o')

    ax[0].legend()
    ax[1].legend()
    fig.savefig("Out.png", dpi=200)