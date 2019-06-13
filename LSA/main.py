# imports

import numpy as np
import re

# Document strings

d1 = """ Romeo and Juliet. """
d2 = """ Juliet: O happy dagger! """
d3 = """ Romeo died by dagger. """
d4 = """ "Live free or die", that's the New-Hampshire's motto. """
d5 = """ Did you know, New-Hampshire is in New-England. """

documents = [d1,d2,d3,d4,d5]

# Pre-process documents

words = set()
for d in documents:
    d = d.lower()
    for word in re.compile(r'\w+').findall(d):
        words.add(word)

for word in list(words):
    if len(word) == 1:
        words.remove(word)

words = list(words)

# Create "A" matrix

A = np.zeros((len(words), len(documents)))

for j, d in enumerate(documents):
    d = d.lower()
    for i, word in enumerate(words):
        counter = 0
        for match in re.compile(word).finditer(d):
            counter += 1

        A[i][j] = counter

B = np.matmul(np.transpose(A), A)
eig_value, eig_vector = np.linalg.eig(B)

sigma = np.identity(len(eig_value))
for i, val in enumerate(eig_value):
    sigma[i][i] = val

U = eig_vector

S = np.zeros(A.shape)
for i in range(len(eig_vector)):
    sig = eig_value[i]
    y = (1/sig)*np.matmul(A, eig_vector[:][i])
    S[:, i] = y

term_concept_matrix = np.matmul(S, sigma)

# Create Search dict

terms = dict()

for i, word in enumerate(words):
    terms[word] = term_concept_matrix[i]

# search query

query = input("Query : ")
query = query.strip().split()

query_vectors = []
for word in query:
    query_vectors.append(terms[word])

query = np.average(np.transpose(query_vectors), axis=1)

# distance calculation

document_concept_matrix = np.matmul(sigma, U)
latent_semantic_score = []

for d in range(document_concept_matrix.shape[1]):
    score = np.dot(document_concept_matrix[:, d], query)
    score = score / (np.linalg.norm(document_concept_matrix[:, d]) * np.linalg.norm(query))
    latent_semantic_score.append(score)

print(latent_semantic_score)
print("\nResults : \n")
for i in np.argsort(latent_semantic_score)[::-1]:
    print("Document {} : ".format(i), documents[i])