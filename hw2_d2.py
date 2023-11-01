import pandas as pd
import numpy as np
import random
from random import randint
import copy
import sklearn
from sklearn.model_selection import train_test_split

# class global variables 
depth = 0
operators = ["+", "-", "/", "*"]
operands = ["x1", "x2", "x3"]
operands = ["-10.0", "-9.0", "-8.0", "-7.0", "-6.0", "-5.0", "-4.0", "-3.0", "-2.0", "-1.0", "0.0", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "x1", "x1", "x1", "x1", "x1", "x1", "x1", "x2", "x2", "x2", "x2", "x2", "x2", "x2", "x3", "x3", "x3", "x3", "x3", "x3", "x3"]


# Implementation global variables 
NUM_GENS = 20
MUTATION_PROB = 0.20
CROSSOVER_PROB = 0.65
#CLONING_PROB = 0.3
POPULATION_SIZE = 500

class Node:
    def __init__(self, v, l, r, truefalse):
        self.left = l
        self.right = r
        self.value = v
        self.operation = truefalse
        
        
class Tree:
    def __init__(self, depth=None, clone=None, fitness=None):
        if clone is None:
            depthList = [0, 1, 2, 3]
            randomList = random.choices(depthList, weights=(1, 2, 3, 1))
            max_depth = randomList[0]
            self.root = self.grow(depth=max_depth)
            self.depth = self.findDepth(self.root)
                
            # controls for bloat
            if (self.depth >= 17):
                self.fitness = float('inf')
            else: 
                self.fitness = self.findFitness()
            
        else:
            self.root = self.clone(clone) #either a root or node
            self.depth = self.findDepth(self.root)
            if (self.depth >= 17):
                self.fitness = float('inf')
            else: 
                self.fitness = self.findFitness()
        
    
    def findFitness(self):
        dataset2 = pd.read_csv("dataset2.csv")
        d2_train, d2_test = train_test_split(dataset2, test_size=0.2, random_state=42, shuffle = False)
        #d2_train = d2_train[:500]
        sum_squared_error = 0
        for i in range(len(d2_train['x1'])):
            if self.evaluate(self.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]) == float('inf'):
                return float('inf')
            if np.isnan(float(self.evaluate(self.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]))):
                return float('inf')
            else:
                sum_squared_error += (self.evaluate(self.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]) - d2_train['f(x1'][i])**2
        mean_squared_error = round(sum_squared_error/len(d2_train['x1']), 20)
        return mean_squared_error
    
    
    def clone(self, node):
        new_node = Node(v=node.value, l=None, r=None, truefalse = node.operation)
        
        if node.left is None and node.right is None:
            return new_node
        
        if node.left is not None:
            new_node.left = self.clone(node.left)
        if node.right is not None:
            new_node.right = self.clone(node.right)
        return new_node

    
    # 0 is operator, 1 is operand
    def grow(self, depth): 
        if (depth == 0):
            is_operand = True
            value = np.random.choice (operands)
            node = Node(v=value, l=None, r=None, truefalse = is_operand)
        else:
            is_operand = False
            value = np.random.choice (operators)   
            node = Node(v=value, l=self.grow(depth-1), r=self.grow(depth-1), truefalse = is_operand)

        return node
        
        
    def findDepth(self, node):
        if node is None:
            return 0
        
        leftDepth = self.findDepth(node.left)
        rightDepth = self.findDepth(node.right)
        
        return max(leftDepth,rightDepth)+1
        
        
    def mutate(self):
        mutated_tree = Tree(clone = self.root)

        current_node = mutated_tree.root
        
        while (not current_node.operation):
            direction_choice = np.random.choice(["left", "right"])
            
            if (direction_choice == "left"):
                current_node = current_node.left
            if (direction_choice == "right"):
                current_node = current_node.right
        
        decision = np.random.choice(["0", "1"])

        #know it's a operand
        if (current_node.value == "x1" or current_node.value == "x2" or current_node.value == "x3"):
            if decision == 0:
                current_node.value = str(round(random.uniform(-10, 10), 2))
            else:
                x_value = current_node.value
                current_node.value = np.random.choice (operators)   
                current_node.operation = False
                current_node.left = Node(v=x_value, l=None, r=None, truefalse = True)
                value = str(round(random.uniform(-10, 10), 2))
                current_node.right = Node(v=value, l=None, r=None, truefalse = True)
        else:
            if decision == 0:
                decision_x = np.random.choice(["0", "1", "2"])
                if decision_x == 0:
                    current_node.value = "x1"
                elif decision_x == 1:
                    current_node.value = "x2"
                else:
                    current_node.value = "x3"
            else:
                current_node.value = float(current_node.value)
                value_change = np.random.choice([-1.00, 1.00])
                current_node.value += value_change
                current_node.value = str(current_node.value)
                
        mutated_tree.depth = mutated_tree.findDepth(mutated_tree.root)

        return mutated_tree
        
        
    def crossover(self, other):
        copied_self = Tree(clone=self.root)
        copied_other = Tree(clone=other.root)
        
        if (copied_self.depth == 0 and copied_other.depth == 0):
            return copied_self
        
        current_self_node = copied_self.root
        current_other_node = copied_other.root
  
        # find crossover point in self
        current_level = 0
        self_level = randint(0, copied_self.depth)
        
        if self_level == 0:
            current_parent_node = current_self_node
        
        if (copied_self.depth == 0):
            current_parent_node = copied_self.root
        else:
            while (current_level < self_level):
                direction_choice_self = np.random.choice(["left", "right"])

                if (direction_choice_self == "left"):
                    current_parent_node = current_self_node
                    current_self_node = current_self_node.left

                if (direction_choice_self == "right"):
                    current_parent_node = current_self_node
                    current_self_node = current_self_node.right              

                if current_self_node is not None:
                    current_level += 1  
                else:
                    current_level = 0
                    current_self_node = copied_self.root
                    self_level = randint(0, copied_self.depth)
                    if self_level == 0:
                        current_parent_node = current_self_node
        
        # find crossover point in other
        current_level = 0
        other_level = randint(0, copied_other.depth)
        if other_level == 0:
            current_other_node = copied_other.root
      
        while (current_level < other_level):
            direction_choice_other = np.random.choice(["left", "right"])

            if (direction_choice_other == "left"):
                current_other_node = current_other_node.left
            if (direction_choice_other == "right"):
                current_other_node = current_other_node.right

            if current_other_node is not None:
                current_level += 1

            else:
                current_level = 0
                current_other_node = copied_other.root
                other_level = randint(0, copied_other.depth)
                if other_level == 0:
                    current_other_node = copied_other.root
        
        if ((copied_self.depth == 0) or (self_level == 0)):
            copied_self.root = current_other_node
            
        elif (direction_choice_self == "left"):
            current_parent_node.left = current_other_node
        else:
            current_parent_node.right = current_other_node
            
        copied_self.depth = self.findDepth(copied_self.root)
        
        return copied_self
    
    
    def evaluate(self, node, x1, x2, x3):
        if node is None:
            return 0
        
        if node.value == "/" and node.right == 0:
            return float('inf')

        # leaf node
        if node.left is None and node.right is None:
            if (node.value == "x1"):
                return float(x1)
            if (node.value == "x2"):
                return float(x2)
            if (node.value == "x3"):
                return float(x3)
            else:
                return float(node.value)

        left_sum = self.evaluate(node.left, x1, x2, x3)

        right_sum = self.evaluate(node.right, x1, x2, x3)

        if node.value == "+":
            return float(left_sum + right_sum)

        elif node.value == "-":
            return float(left_sum - right_sum)
        
        elif node.value == "/":
            if (right_sum == 0) and (left_sum == 0):
                return float('inf')
            elif (right_sum == 0):
                return float('inf')
            else: 
                return float(left_sum / right_sum)
        
        elif node.value == "*":
            return float(left_sum * right_sum) 
        
        
        
    def postOrderIterative(self):
        if self.root is None: 
            return        
      
        s1 = [] 
        s2 = [] 
      
        s1.append(self.root) 
        while s1: 
          
        # Pop an item from s1 and  
        # append it to s2 
            node = s1.pop() 
            s2.append(node) 
      
        # Push left and right children of  
        # removed item to s1 
            if node.left: 
                s1.append(node.left) 
            if node.right: 
                s1.append(node.right) 
  
        while s2: 
            node = s2.pop()
            print(node.value,end=" ")
            

# IMPLEMENTATION
def initialize_population(size):
    current_gen = []
    for i in range(size):
        current_gen.append(Tree())
    return current_gen

def fitness_to_weights(current_gen):
    fitnesses = []
    for t in current_gen:
        fitnesses.append(t.fitness)
    largest = max(fitnesses)
    total = sum(fitnesses)
    weights = [largest - f for f in fitnesses]
    return weights

def inOrder(node):
    if node is not None:
        print("(", end = '')
        inOrder(node.left)
        print(node.value, end = '')
        inOrder(node.right)
        print(")", end = '')
    return         

def tournament (current_gen, size):
    index = randint(0, len(current_gen)-1)
    while current_gen[index].fitness == float('inf'):
        index = randint(0, (len(current_gen)-1))
    
    best_fitness = current_gen[index].fitness
    best_tree = copy.deepcopy(current_gen[index])
    
    
    for i in range(size):
        index = randint(0, (len(current_gen)-1))
        if current_gen[index].fitness < best_fitness:
            best_fitness = current_gen[index].fitness
            best_tree = copy.deepcopy(current_gen[index])       
    return best_tree    

def roulette_wheel_selection(current_gen):
    fitness_scores = []
    for tree in range(len(current_gen)):
        fitness_scores.append(current_gen[tree].fitness)
        
    total_fitness = sum(fitness_scores)
    
    spin = random.uniform(0, total_fitness)
    
    cumulative_fitness = 0
    selected_individual = None
    
    # Iterate through the population and select an individual
    for i in range(len(current_gen)):
        cumulative_fitness += total_fitness - fitness_scores[i]
        if cumulative_fitness >= spin:
            selected_individual = current_gen[i]
            break
    
    return selected_individual

def find_fittest (current_gen):
    fittest = float('inf')
    tree = None

    for t in current_gen:
        if t.fitness < fittest:
            tree = t
            fittest = t.fitness
    return tree

def findTestFitness(tree):
        dataset2 = pd.read_csv("dataset2.csv")
        d2_train, d2_test = train_test_split(dataset2, test_size=0.2, random_state=42, shuffle = False)
        d2_test = d2_test.reset_index()
        sum_squared_error = 0
        for i in range(len(d2_test['x1'])):
            if tree.evaluate(tree.root, d2_test['x1'][i], d2_test['x2'][i], d2_test['x3'][i]) == float('inf'):
                return float('inf')
            if np.isnan(float(tree.evaluate(tree.root, d2_test['x1'][i], d2_test['x2'][i], d2_test['x3'][i]))):
                return float('inf')
            else:
                sum_squared_error += (tree.evaluate(tree.root, d2_test['x1'][i], d2_test['x2'][i], d2_test['x3'][i]) - d2_test['f(x1'][i])**2
        mean_squared_error = round(sum_squared_error/len(d2_test['x1']), 20)
        return mean_squared_error
    
    
def findFinalFitness(tree):
        dataset2 = pd.read_csv("dataset2.csv")
        d2_train, d2_test = train_test_split(dataset2, test_size=0.2, random_state=42, shuffle = False)
        d2_test = d2_test.reset_index()
        sum_squared_error = 0
        for i in range(len(d2_train['x1'])):
            if tree.evaluate(tree.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]) == float('inf'):
                return float('inf')
            if np.isnan(float(tree.evaluate(tree.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]))):
                return float('inf')
            else:
                sum_squared_error += (tree.evaluate(tree.root, d2_train['x1'][i], d2_train['x2'][i], d2_train['x3'][i]) - d2_train['f(x1'][i])**2
        mean_squared_error = round(sum_squared_error/len(d2_train['x1']), 20)
        return mean_squared_error
    
        
def end_result(NUM_GENS, POPULATION_SIZE, MUTATION_PROB, current_gen, best, best_fitnesses, i):
        
    for i in range(NUM_GENS):
        next_gen = []

        # populate next generation
        for j in range(POPULATION_SIZE):
            selection = np.random.choice(["0", "1"])
            if selection == "0":
                parent1 = tournament(current_gen, 50)
            else:
                parent1 = roulette_wheel_selection(current_gen)

            flip = random.random()
            if flip <= MUTATION_PROB:
                # mutation
                next_gen.append(parent1.mutate())
            elif flip <= CROSSOVER_PROB + MUTATION_PROB:
                # crossover
                selection = np.random.choice(["0", "1"])
                if selection == "0":
                    parent2 = tournament(current_gen, 50)
                else:
                    parent2 = roulette_wheel_selection(current_gen)
                next_gen.append(parent1.crossover(parent2))
            else:  
                # clone
                next_gen.append(parent1)
            
    # remember the best individual in this generation
        best_tree = min(current_gen, key=lambda t: t.fitness)
        best.append(best_tree)
        
        print ("generation: ", i)
        inOrder(best_tree.root)
        print()
        print("best tree fitness: ", best_tree.fitness)
        best_fitnesses.append(best_tree.fitness)
        
        if i>3:
            if ((best_fitnesses[i] == best_fitnesses[i-1]) and (best_fitnesses[i] == best_fitnesses[i-2]) and (best_fitnesses[i] == best_fitnesses[i-3])):
                MUTATION_PROB = MUTATION_PROB*1.2
        
        # reset current gen
        current_gen = []
        for t in next_gen:
            current_gen.append(t)
            
    return best_tree

def main():
    fittest_trees = []
    for j in range(10): 
        print ("evolutionary run number: ", j)
        current_gen = initialize_population(POPULATION_SIZE)
        fittest_tree = end_result(NUM_GENS, POPULATION_SIZE, MUTATION_PROB, current_gen, [], [], 0)
        print("fittest tree in this run is: ")
        inOrder(fittest_tree.root)
        print()
        print("its fitness on the training set is: ", fittest_tree.fitness)
        #print("its fitness on the test set is: ", findTestFitness(best_tree))
        fittest_trees.append(fittest_tree)

    # find best tree across all 40 evolutions 
    best_fitness = float('inf')
    best_tree = None
    for i in range(len(fittest_trees)):
        if fittest_trees[i].fitness < best_fitness:
            best_fitness = fittest_trees[i].fitness
            best_tree = fittest_trees[i]

    # run best tree on entire dataset
    print("Reached end of evolutions!")
    print("fittest tree overall is: ")
    inOrder(best_tree.root)
    print()
    print("its fitness on the training set: ", findFinalFitness(best_tree))
    print("its fitness on the test set is: ", findTestFitness(best_tree))

    
if __name__ == "__main__":
    main()
    
    