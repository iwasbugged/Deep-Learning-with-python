import numpy as np

samples = ['The cat sat on the mat.', ' The dog ate myy homework.']

""" Intial data :  one entry per sample (In this example a sample is a sentence , but it could be an entire
document)
"""
token_index = {}  
""" Builds an index of all tokens in the data"""
for sample in samples:
    for word in sample.split():
        
        """Tokenizes the sample via the split method. In real life, we'd also strip puntuation and special 
        characters from the samples.
        """

        if word not in token_index:
            token_index[word] = len(token_index) + 1
            
            """ Assigns a unique index to each unique word. Note that we don't attribute index '0' to anything
            """

max_length = 10

"""Vectorizes the samples , you'll only consider the first max_length (In this case 10) word in each sample
"""
results = np.zeros (shape=( len(samples),
                    max_length,
                    max(token_index.values()) + 1 ))

""" This where we store the results.
"""
for i , sample in enumerate(samples):
    for j , word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i , j , index] = 1

print(results)

        