import random
import numpy as np
import pandas as pd

class Node:
    
    def __init__(self, name, value, children, parent, player_to_choose, visited, utility_player1, utility_player2):
        self.name = name
        self.value = value
        self.children = children
        self.parent = parent
        self.player_to_choose = player_to_choose
        self.visited = visited
        self.utility_player1 = utility_player1
        self.utility_player2 = utility_player2
        self.action = None
        self.depth = 0
        # self.edges = [left_prob, right_prob]
    
    def get_action(self):
        return self.action
    
    def set_action(self, action):
        self.action = action
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1
        
    def set_children(self, children):
        self.children = children

    def get_leaf_nodes(self):
        leaf_nodes = []
        for child in self.children:
            leaf_nodes += child.get_leaf_nodes()
        return leaf_nodes
        
    def set_player_to_choose(self, player_to_choose):
        self.player_to_choose = player_to_choose
        
    def get_player_to_choose(self):
        return self.player_to_choose
        
    def set_value(self, value):
        self.value = value
        
    def set_parent(self, parent):
        self.parent = parent
        
    def set_name(self, name):
        self.name = name
        
    def set_utility_player1(self, utility_player1):
        self.utility_player1 = utility_player1
        
    def set_utility_player2(self, utility_player2):
        self.utility_player2 = utility_player2
                
    def set_utility(self, utility_player1, utility_player2):
        self.utility_player1 = utility_player1
        self.utility_player2 = utility_player2
    
    def get_parent(self):
        return self.parent
    
    def get_child(self, index):
        return self.children[index]
    
    def set_visited(self):
        self.visited = True
        
    def get_visited(self):
        return self.visited
    
    def get_utility(self):
        return self.utility_player1, self.utility_player2
    
    def get_name(self):
        return self.name
    
    def get_value(self):
        return self.value
    
    def get_children(self):
        return self.children
    
    def remove_child(self, child):
        self.children.remove(child)
        
    def show_info(self):
        print("Name: ", self.name)
        print("Value: ", self.value)
        print("Children: {[ c.name  for c in self.children)]}")
        print("Parent: ", self.parent)
        print("Player to choose: ", self.player_to_choose)
        print("Visited: ", self.visited)
        print("Utility: ", self.utility_player1, self.utility_player2)
        print("Action: ", self.action)
        print("Depth: ", self.depth)
