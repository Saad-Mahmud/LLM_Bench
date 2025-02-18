import numpy as np
from collections import defaultdict
def calculate_entropy(logits):
    """
    Calculate the entropy of a distribution given an array of negative log likelihoods (logits).

    :param logits: Array of negative log likelihoods.
    :return: Entropy value.
    """
    # Convert NLL (negative log likelihood) values to probabilities
    probabilities = np.exp(-np.array(logits))
    # Normalize the probabilities
    probabilities /= probabilities.sum()
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))  # Add epsilon to avoid log(0)
    return entropy

def find_lowest_entropy_object(data_objects):
    """
    Find the object with the lowest entropy in the 'logits' distribution.

    :param data_objects: List of objects, each containing an array of negative log likelihoods under 'logits' key.
    :return: The object with the lowest entropy in the 'logits' distribution.
    """
    # Initialize variables to track the minimum entropy and corresponding object
    min_entropy = float('inf')
    min_entropy_object = None
    
    # Iterate over each object to calculate entropy
    for obj in data_objects:
        entropy = calculate_entropy(obj.json()["logits"])
        
        # Update if current object has lower entropy
        if entropy < min_entropy:
            min_entropy = entropy
            min_entropy_object = obj

    return min_entropy_object

def find_majority_object(data_objects):
    """
    Find the object with the lowest entropy in the 'logits' distribution.

    :param data_objects: List of objects, each containing an array of negative log likelihoods under 'logits' key.
    :return: The object with the lowest entropy in the 'logits' distribution.
    """
    # Initialize variables to track the minimum entropy and corresponding object
    data = defaultdict(lambda: 0)
    
    # Iterate over each object to calculate entropy
    for obj in data_objects:
        data[obj.json()["output"]]+=1
    
    data = sorted([(-data[v],v) for v in data])
    for obj in data_objects:
        if obj.json()["output"] == data[0][1]:
            return obj
    
    