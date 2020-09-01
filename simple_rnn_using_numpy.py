import numpy as np

timesteps = 100  
''' Number of timesteps in the input'''

input_features = 32
output_features = 64
''' Dimensionality of the input/output feature space'''

inputs = np.random.random((timesteps , input_features))
''' Input data : random noise for the sake of example'''
state_t = np.zeros((output_features,))
''' Initial state : An all zeros vector'''

''' Creating random weights matrics'''
w = np.random.random((output_features , input_features))
u = np.random.random((output_features , input_features))
b = np.random.random((output_features , input_features))

successive_outputs = []
''' input_t is a vector of shape(input_features)'''
for input_t in inputs:
    output_t = np.tanh(np.dot(w , input_t) + np.dot(u , state_t) + b)
    ''' 
    Combines the input with the current state (The previous output) to obtain the current output
    '''
    successive_outputs.append(output_t)
    '''
    Stores this output in a list
    '''
    state_t = output_t
    ''' 
    Update the state of the network for the next timesteps
    '''

final_output_sequence = np.concatenate( successive_outputs , axis=0)

'''
The final output is 2D tensor of the shape (timesteps , output_features)
'''

print(final_output_sequence)
