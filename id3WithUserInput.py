import pandas as pd
import math
from collections import Counter
from pprint import pprint
from graphviz import Digraph  # Importing the graphviz module


dataset_path = input("Enter the path of the Dataset CSV file: ")        

# Read the dataset from a CSV file
df = pd.read_csv(dataset_path)
print("\n Given Dataset", df)

# Identify the target attribute
target_attribute = df.keys()[-1]
print('Target Attribute is   : ', target_attribute)

# Get the attribute names from the input dataset
attribute_names = list(df.keys())

# Remove the target attribute from the attribute names list
attribute_names.remove(target_attribute) 
print('Predicting Attributes : ', attribute_names)

# Function to calculate the entropy of probability of observations
def entropy(probs):  
    return sum([-prob * math.log(prob, 2) for prob in probs])

# Function to calculate the entropy of the given dataset/list with respect to target attributes
def entropy_of_list(ls, value):  
    # Total instances associated with the respective attribute
    total_instances = len(ls)
    print("---------------------------------------------------------")
    print("\nTotal no of instances/records associated with '{0}' = {1}".format(value, total_instances))
    
    # Counter calculates the proportion of class
    cnt = Counter(x for x in ls)
    print('\nTarget attribute class count(Yes/No) =', dict(cnt))
    
    # x means the number of YES/NO
    probs = [x / total_instances for x in cnt.values()]  
    print("\nClasses : ", max(cnt), min(cnt))

    prob_positive_calculation = "{0} / {1}".format(cnt[max(cnt)], total_instances)
    prob_negative_calculation = "{0} / {1}".format(cnt[min(cnt)], total_instances)

    print("\nProbabilities of Class 'Positive': '{0}' = {1} = {2}".format(max(cnt), prob_positive_calculation, max(probs)))
    print("Probabilities of Class 'Negative': '{0}'  = {1} = {2}".format(min(cnt), prob_negative_calculation, min(probs)))

    print("Entropy ({0}) = -{1} * log2 * {1} - {2} * log2 * {2} = {3}".format(value,max(probs), min(probs), entropy(probs)))

    # Call Entropy 
    return entropy(probs) 

# Function to calculate the information gain for a specific attribute
def information_gain(df, split_attribute, target_attribute, battr):
    print("\n\n----- Information Gain Calculation of", split_attribute, "----- ") 
    
    # Group the data based on attribute values
    df_split = df.groupby(split_attribute) 
    glist = []
    for gname, group in df_split:
        print('Grouped Attribute Values \n', group)
        print("---------------------------------------------------------")
        glist.append(gname) 
    
    glist.reverse()
    nobs = len(df.index) * 1.0   
    df_agg1 = df_split.agg({target_attribute: lambda x: entropy_of_list(x, glist.pop())})
    df_agg2 = df_split.agg({target_attribute: lambda x: len(x) / nobs})
    
    df_agg1.columns = ['Entropy']
    df_agg2.columns = ['Proportion']

    # Calculate Information Gain:
    new_entropy = sum(df_agg1['Entropy'] * df_agg2['Proportion'])
    if battr != 'S':
        old_entropy = entropy_of_list(df[target_attribute], 'S-' + str(df.iloc[0][battr]))
    else:
        old_entropy = entropy_of_list(df[target_attribute], battr)

    print("Entropy ({0}) = {1}".format(split_attribute,new_entropy))
    print("Information Gain = {0} - {1}".format(old_entropy, new_entropy))
    return old_entropy - new_entropy

# Function to build the ID3 decision tree
def id3(df, target_attribute, attribute_names, default_class=None, default_attr='S'):
    cnt = Counter(x for x in df[target_attribute])  # class of YES/NO
    
    # First check: Is this split of the dataset homogeneous?
    if len(cnt) == 1:
        return next(iter(cnt))  # next input data set, or raises StopIteration when EOF is hit.
    
    # Second check: Is this split of the dataset empty? If yes, return a default value
    elif df.empty or (not attribute_names):
        return default_class  # Return None for an Empty Data Set
    
    # Otherwise: This dataset is ready to be divided up!
    else:
        # Get Default Value for the next recursive call of this function:
        default_class = max(cnt.keys())  # No of YES and NO Class
        # Compute the Information Gain of the attributes:
        gainz = []
        for attr in attribute_names:
            ig = information_gain(df, attr, target_attribute, default_attr)
            gainz.append(ig)
            print('\nInformation gain of', '“', attr, '”', ': ', ig)
            print("=========================================================")
        
        index_of_max = gainz.index(max(gainz))               # Index of the Best Attribute
        best_attr = attribute_names[index_of_max]            # Choose Best Attribute to split on
        print("\nList of Gain for attributes:", attribute_names, "\nare:", gainz, "respectively.")
        print("\nAttribute with the maximum gain is : ", best_attr)
        print("\nHence, the Root node will be : ", best_attr)
        print("=========================================================")

        # Create an empty tree, to be populated in a moment
        tree = {best_attr: {}}  # Initiate the tree with the best attribute as a node 
        remaining_attribute_names = [i for i in attribute_names if i != best_attr]
        
        # Split the dataset - On each split, recursively call this algorithm.
        # Populate the empty tree with subtrees, which are the result of the recursive call
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset, target_attribute, remaining_attribute_names, default_class, best_attr)
            tree[best_attr][attr_val] = subtree
        return tree

# Function to calculate the entropy of the given dataset with respect to target attributes
def entropy_dataset(a_list):  
    cnt = Counter(x for x in a_list)  # Counter calculates the proportion of class
    num_instances = len(a_list) * 1.0    # = 14
    print("\nNumber of Instances of the Current Sub-Class = {0}".format(num_instances))
    
    # x means the number of YES/NO
    probs = [x / num_instances for x in cnt.values()]  
    print("Classes: ", "'Positive' =", max(cnt), ",'Negative' =", min(cnt))

    # Print the equation for probabilities
    print("\nEquation for Probabilities:")

    prob_positive_calculation = "{0} / {1}".format(cnt[max(cnt)], num_instances)
    prob_negative_calculation = "{0} / {1}".format(cnt[min(cnt)], num_instances)
    
    print("Probabilities of Class 'Positive': {0} = {1} = {2}".format(max(cnt), prob_positive_calculation, max(probs)))
    print("Probabilities of Class 'Negative': {0} = {1} = {2}".format(min(cnt), prob_negative_calculation, min(probs)))

    print("Entropy = -{0} * log2 * {0} - {1} * log2 * {1}".format(max(probs), min(probs)))

    # Call Entropy
    return entropy(probs) 

def visualize_tree(tree, parent_name, graph):
    for key, value in tree.items():
        if isinstance(value, dict):
            # Recursively visualize subtrees
            visualize_tree(value, f'{parent_name}_{key}', graph)
        else:
            # Leaf node
            graph.node(f'{parent_name}_{key}', label=f'{key}\n{value}', shape='box')

        if parent_name is not None:
            # Connect parent node to current node
            graph.edge(parent_name, f'{parent_name}_{key}')

# The initial entropy of the YES/NO attribute for our dataset
print("\nEntropy calculation for the input dataset:")
print(df['PlayTennis'])

total_entropy = entropy_dataset(df['PlayTennis'])
print("\nTotal Entropy(S) of PlayTennis Dataset = ", total_entropy)
print("=========================================================")

# Build the ID3 decision tree
tree = id3(df, target_attribute, attribute_names)
print("\nThe Resultant Decision Tree is: \n")
pprint(tree)

attribute = next(iter(tree))
print("\nBest Attribute = ", attribute)
print("Tree Keys      = ", tree[attribute].keys())

# Create a graph using graphviz
dot = Digraph(comment='Decision Tree')

# Visualize the decision tree
visualize_tree(tree, 'Root', dot)

# Save the graph as a PNG file
dot.render('decision_tree', format='png', cleanup=True)
