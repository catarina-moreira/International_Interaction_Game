import pyAgrum as gum
from pylab import *
import pyAgrum.lib.notebook as gnb

from node import Node

import numpy as np
import pandas as pd
import random

class Agent:

    def __init__(self, name, id, utilities):
        self.name = name                # name of the country
        self.id = id                    # checks whether the agent is the first player (id = 1) or the second player (id = 2)
        self.actions = [                # the actions that the agent can take
            "apply force",      
            "do nothing"]
        self.current_action = None      # the action that the agent is currently taking
        self.history = []               # keeping track of the actions taken by the agent [for debugging purposes]
        self.current_node = -1          # keeping track of where the agent is in the tree
        self.start_node = -1            # the node where the agent starts
        self.utilities = utilities
        self.game_ended = "NO"
        
    def print_info(self):
        print(f"Agent {self.name} with id {self.id}")
        print(f"Actions: {self.actions}")
        print(f"Current action: {self.current_action}")
        print(f"History: {self.history}")
        print(f"Current node: {self.current_node}")
        print(f"Start node: {self.start_node}")
        print(f"Utilities: {self.utilities}")
        print(f"Game ended: {self.game_ended}")
        print("\n")
    
    
    def set_current_action(self, action):
        self.current_action = action
    
    def get_current_action(self):
        return self.current_action
    
    def set_history(self, history):
        self.history = history
    
    def get_history(self):
        return self.history
    
    def set_current_node(self, current_node):
        self.current_node = current_node
    
    def get_current_node(self): 
        return self.current_node
    
    def set_start_node(self, start_node):
        self.start_node = start_node
        
    def get_start_node(self):
        return self.start_node
    
    def set_utilities(self, utilities):
        self.utilities = utilities
        
    def get_utilities(self):
        return self.utilities
    
    def set_game_ended(self, game_ended):
        self.game_ended = game_ended
    
    def get_game_ended(self):
        return self.game_ended
    
    def set_name(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def set_id(self, id):
        self.id = id
        
    def get_id(self):
        return self.id
    
    
    def erase_arc(self, node1, node2, tree):
        try:
            tree.eraseArc(node1,node2)
        except Exception:
            print(f"ARC {node1} -> {node2} does not exist...")
        return tree
        
    
    def perspective_player1(self, other_agent, game_tree, game_tree_dict):
        
        PLAYER_1 = 0
        PLAYER_2 = 1
        # start the backward induction if it is player 1's move
        # NODE 12 PLAYER 1: CAP1 vs WAR2
        if self.utilities["CAP1"] > self.utilities["WAR2"]:
            game_tree_dict["12"].set_utility_player1( self.utilities["CAP1"] )
            game_tree_dict["12"].set_utility_player2( other_agent.utilities["CAP1"] )
            game_tree = self.erase_arc("12", "WAR2", game_tree)
            game_tree_dict["12"].set_children([game_tree_dict["CAP1"]])
            game_tree_dict["12"].set_action("do nothing")
        else:
            game_tree_dict["12"].set_utility_player1( self.utilities["WAR2"] )
            game_tree_dict["12"].set_utility_player2( other_agent.utilities["WAR2"] )
            game_tree = self.erase_arc("12", "CAP1", game_tree)
            game_tree_dict["12"].set_children([game_tree_dict["WAR2"]])
            game_tree_dict["12"].set_action("apply force")
            
        # NODE 9 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if other_agent.utilities["NEG"] > game_tree_dict["12"].get_utility()[PLAYER_2]:
            game_tree_dict["9"].set_utility_player1(self.utilities["NEG"])
            game_tree_dict["9"].set_utility_player2(other_agent.utilities["NEG"])
            game_tree = self.erase_arc("9", "12", game_tree)
            game_tree_dict["9"].set_children([game_tree_dict["NEG"]])
            game_tree_dict["9"].set_action("do nothing")
        else:
            game_tree_dict["9"].set_utility_player1(  game_tree_dict["12"].get_utility()[PLAYER_1] )
            game_tree_dict["9"].set_utility_player2(  game_tree_dict["12"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("9", "NEG", game_tree)
            game_tree_dict["9"].set_children([game_tree_dict["12"]])
            game_tree_dict["9"].set_action("apply force")
            
        # NODE 10 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if other_agent.utilities["CAP2"] > other_agent.utilities["WAR1"]:
            game_tree_dict["10"].set_utility_player1(self.utilities["CAP2"])
            game_tree_dict["10"].set_utility_player2(other_agent.utilities["CAP2"])
            game_tree = self.erase_arc("10", "WAR1", game_tree)
            game_tree_dict["10"].set_children([game_tree_dict["CAP2"]])
            game_tree_dict["10"].set_action("do nothing")
        else:
            game_tree_dict["10"].set_utility_player1(self.utilities["WAR1"])
            game_tree_dict["10"].set_utility_player2(other_agent.utilities["WAR1"])
            game_tree = self.erase_arc("10", "CAP2", game_tree)
            game_tree_dict["10"].set_children([game_tree_dict["WAR1"]])
            game_tree_dict["10"].set_action("apply force")
            
        # NODE 5 PLAYER 1: NODE 9 vs NODE 10
        if game_tree_dict["9"].get_utility()[PLAYER_1] > game_tree_dict["10"].get_utility()[PLAYER_1]:
            game_tree_dict["5"].set_utility_player1( game_tree_dict["9"].get_utility()[PLAYER_1] )
            game_tree_dict["5"].set_utility_player2( game_tree_dict["9"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("5", "10", game_tree)
            game_tree_dict["5"].set_children([game_tree_dict["9"]])
            game_tree_dict["5"].set_action("do nothing")
        else:
            game_tree_dict["5"].set_utility_player1( game_tree_dict["10"].get_utility()[PLAYER_1])
            game_tree_dict["5"].set_utility_player2( game_tree_dict["10"].get_utility()[PLAYER_2])
            game_tree = self.erase_arc("5", "9", game_tree)
            game_tree_dict["5"].set_children([game_tree_dict["10"]])
            game_tree_dict["5"].set_action("apply force")
            
        # NODE 3 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 3
        if other_agent.utilities["ACQ2"] > game_tree_dict["5"].get_utility()[PLAYER_2]:
            game_tree_dict["3"].set_utility_player1( self.utilities["ACQ2"] )
            game_tree_dict["3"].set_utility_player2( other_agent.utilities["ACQ2"])
            game_tree = self.erase_arc("3", "5", game_tree)
            game_tree_dict["3"].set_children([game_tree_dict["ACQ2"]])
            game_tree_dict["3"].set_action("do nothing")
        else:
            game_tree_dict["3"].set_utility_player1( game_tree_dict["5"].get_utility()[PLAYER_1] )
            game_tree_dict["3"].set_utility_player2( game_tree_dict["5"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("3", "ACQ2", game_tree)
            game_tree_dict["3"].set_children([game_tree_dict["5"]])
            game_tree_dict["3"].set_action("apply force")
        
        # NODE 11 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if other_agent.utilities["CAP2"] > other_agent.utilities["WAR1"]:
            game_tree_dict["11"].set_utility_player1( self.utilities["CAP2"] )
            game_tree_dict["11"].set_utility_player2( other_agent.utilities["CAP2"])
            game_tree = self.erase_arc("11", "WAR1", game_tree)
            game_tree_dict["11"].set_children([game_tree_dict["CAP2"]])
            game_tree_dict["11"].set_action("do nothing")
        else: 
            game_tree_dict["11"].set_utility_player1( self.utilities["WAR1"] )
            game_tree_dict["11"].set_utility_player2( other_agent.utilities["WAR1"])
            game_tree = self.erase_arc("11", "CAP2", game_tree)
            game_tree_dict["11"].set_children([game_tree_dict["WAR1"]])
            game_tree_dict["11"].set_action("apply force")
            
        # NODE 7 PLAYER 1: NEG VS NODE 11
        if self.utilities["NEG"] > game_tree_dict["11"].get_utility()[PLAYER_1]:
            game_tree_dict["7"].set_utility_player1( self.utilities["NEG"] )
            game_tree_dict["7"].set_utility_player2( other_agent.utilities["NEG"] )
            game_tree = self.erase_arc("7", "11", game_tree)
            game_tree_dict["7"].set_children([game_tree_dict["NEG"]])
            game_tree_dict["7"].set_action("do nothing")
        else:
            game_tree_dict["7"].set_utility_player1( game_tree_dict["11"].get_utility()[PLAYER_1])
            game_tree_dict["7"].set_utility_player2( game_tree_dict["11"].get_utility()[PLAYER_2])
            game_tree = self.erase_arc("7", "NEG", game_tree)
            game_tree_dict["7"].set_children([game_tree_dict["11"]])
            game_tree_dict["7"].set_action("apply force")
        
        # NODE 8 
        if self.utilities["CAP1"] > self.utilities["WAR2"]:
            game_tree_dict["8"].set_utility_player1( self.utilities["CAP1"] )
            game_tree_dict["8"].set_utility_player2( other_agent.utilities["CAP1"] )
            game_tree = self.erase_arc("8", "WAR2", game_tree)
            game_tree_dict["8"].set_children([game_tree_dict["CAP1"]])
            game_tree_dict["8"].set_action("do nothing")
        else:
            game_tree_dict["8"].set_utility_player1( self.utilities["WAR2"] )
            game_tree_dict["8"].set_utility_player2( other_agent.utilities["WAR2"] )
            game_tree = self.erase_arc("8", "CAP1", game_tree)
            game_tree_dict["8"].set_children([game_tree_dict["WAR2"]])
            game_tree_dict["8"].set_action("apply force")
        
        # NODE 6 PLAYER 1 IN THER PERSPECTIVE OF PLAYER 2
        if game_tree_dict["7"].get_utility()[PLAYER_2] > game_tree_dict["8"].get_utility()[PLAYER_2]:
            game_tree_dict["6"].set_utility_player1( game_tree_dict["7"].get_utility()[PLAYER_1] )
            game_tree_dict["6"].set_utility_player2( game_tree_dict["7"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("6", "8", game_tree)
            game_tree_dict["6"].set_children([game_tree_dict["7"]])
            game_tree_dict["6"].set_action("do nothing")
        else:
            game_tree_dict["6"].set_utility_player1( game_tree_dict["8"].get_utility()[PLAYER_1] )
            game_tree_dict["6"].set_utility_player2( game_tree_dict["8"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("6", "7", game_tree)
            game_tree_dict["6"].set_children([game_tree_dict["8"]])
            game_tree_dict["6"].set_action("apply force")
        
        # NODE 4
        if self.utilities["ACQ1"] > game_tree_dict["6"].get_utility()[PLAYER_1]:
            game_tree_dict["4"].set_utility_player1( self.utilities["ACQ1"] )
            game_tree_dict["4"].set_utility_player2( other_agent.utilities["ACQ1"] )
            game_tree = self.erase_arc("4", "6", game_tree)
            game_tree_dict["4"].set_action("do nothing")
        else:
            game_tree_dict["4"].set_utility_player1( game_tree_dict["6"].get_utility()[PLAYER_1] )
            game_tree_dict["4"].set_utility_player2( game_tree_dict["6"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("4", "ACQ1", game_tree)
            game_tree_dict["4"].set_children([game_tree_dict["6"]])
            game_tree_dict["4"].set_action("apply force")
        
        # NODE 2
        if other_agent.utilities["SQ"] > game_tree_dict["4"].get_utility()[PLAYER_2]:
            game_tree_dict["2"].set_utility_player1( self.utilities["SQ"] )
            game_tree_dict["2"].set_utility_player2( other_agent.utilities["SQ"] )
            game_tree = self.erase_arc("2", "4", game_tree)
            game_tree_dict["2"].set_children([game_tree_dict["SQ"]])
            game_tree_dict["2"].set_action("do nothing")
        else:
            game_tree_dict["2"].set_utility_player1( game_tree_dict["4"].get_utility()[PLAYER_1] )
            game_tree_dict["2"].set_utility_player2( game_tree_dict["4"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("2", "SQ", game_tree)
            game_tree_dict["2"].set_children([game_tree_dict["4"]])
            game_tree_dict["2"].set_action("apply force")
            
        # NODE 1
        if game_tree_dict["2"].get_utility()[PLAYER_1] > game_tree_dict["3"].get_utility()[PLAYER_1]:
            game_tree_dict["1"].set_utility_player1( game_tree_dict["2"].get_utility()[PLAYER_1] )
            game_tree_dict["1"].set_utility_player2( game_tree_dict["2"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("1", "3", game_tree)
            game_tree_dict["1"].set_children([game_tree_dict["2"]])
            game_tree_dict["1"].set_action("do nothing")
        else:
            game_tree_dict["1"].set_utility_player1( game_tree_dict["3"].get_utility()[PLAYER_1] )
            game_tree_dict["1"].set_utility_player2( game_tree_dict["3"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("1", "2", game_tree)
            game_tree_dict["1"].set_children([game_tree_dict["3"]])
            game_tree_dict["1"].set_action("apply force")
        
        return game_tree, game_tree_dict
    
    def perspective_player2(self, other_agent, game_tree, game_tree_dict):
        
        PLAYER_1 = 0
        PLAYER_2 = 1
        
        # start the backward induction if it is player 1's move
        # NODE 12 PLAYER 1: CAP1 vs WAR2
        if other_agent.utilities["CAP1"] > other_agent.utilities["WAR2"]:
            game_tree_dict["12"].set_utility_player1( other_agent.utilities["CAP1"] )
            game_tree_dict["12"].set_utility_player2( self.utilities["CAP1"] )
            game_tree = self.erase_arc("12", "WAR2", game_tree)
            game_tree_dict["12"].set_children([game_tree_dict["CAP1"]])
        else:
            game_tree_dict["12"].set_utility_player1( other_agent.utilities["WAR2"] )
            game_tree_dict["12"].set_utility_player2( self.utilities["WAR2"] )
            game_tree = self.erase_arc("12", "CAP1", game_tree)
            game_tree_dict["12"].set_children([game_tree_dict["WAR2"]])
            
        # NODE 9 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if self.utilities["NEG"] > game_tree_dict["12"].get_utility()[PLAYER_1]:
            game_tree_dict["9"].set_utility_player1(other_agent.utilities["NEG"])
            game_tree_dict["9"].set_utility_player2(self.utilities["NEG"])
            game_tree = self.erase_arc("9", "12", game_tree)
            game_tree_dict["9"].set_children([game_tree_dict["NEG"]])
        else:
            game_tree_dict["9"].set_utility_player1(  game_tree_dict["12"].get_utility()[PLAYER_1] )
            game_tree_dict["9"].set_utility_player2(  game_tree_dict["12"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("9", "NEG", game_tree)
            game_tree_dict["9"].set_children([game_tree_dict["12"]])
            
        # NODE 10 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if self.utilities["CAP2"] > self.utilities["WAR1"]:
            game_tree_dict["10"].set_utility_player1(other_agent.utilities["CAP2"])
            game_tree_dict["10"].set_utility_player2(self.utilities["CAP2"])
            game_tree = self.erase_arc("10", "WAR1", game_tree)
            game_tree_dict["10"].set_children([game_tree_dict["CAP2"]])
        else:
            game_tree_dict["10"].set_utility_player1(other_agent.utilities["WAR1"])
            game_tree_dict["10"].set_utility_player2(self.utilities["WAR1"])
            game_tree = self.erase_arc("10", "CAP2", game_tree)
            game_tree_dict["10"].set_children([game_tree_dict["WAR1"]])
            
        # NODE 5 PLAYER 1: NODE 9 vs NODE 10
        if game_tree_dict["9"].get_utility()[PLAYER_1] > game_tree_dict["10"].get_utility()[PLAYER_1]:
            game_tree_dict["5"].set_utility_player1( game_tree_dict["9"].get_utility()[PLAYER_1] )
            game_tree_dict["5"].set_utility_player2( game_tree_dict["9"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("5", "10", game_tree)
            game_tree_dict["5"].set_children([game_tree_dict["9"]])
        else:
            game_tree_dict["5"].set_utility_player1( game_tree_dict["10"].get_utility()[PLAYER_1])
            game_tree_dict["5"].set_utility_player2( game_tree_dict["10"].get_utility()[PLAYER_2])
            game_tree = self.erase_arc("5", "9", game_tree)
            game_tree_dict["5"].set_children([game_tree_dict["10"]])
            
        # NODE 3 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 3
        if self.utilities["ACQ2"] > game_tree_dict["5"].get_utility()[PLAYER_1]:
            game_tree_dict["3"].set_utility_player1( self.utilities["ACQ2"] )
            game_tree_dict["3"].set_utility_player2( other_agent.utilities["ACQ2"])
            game_tree = self.erase_arc("3", "5", game_tree)
            game_tree_dict["3"].set_children([game_tree_dict["ACQ2"]])
        else:
            game_tree_dict["3"].set_utility_player1( game_tree_dict["5"].get_utility()[PLAYER_1] )
            game_tree_dict["3"].set_utility_player2( game_tree_dict["5"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("3", "ACQ2", game_tree)
            game_tree_dict["3"].set_children([game_tree_dict["5"]])
        
        # NODE 11 PLAYER 1 IN THE PERSPECTIVE OF PLAYER 2
        if self.utilities["CAP2"] > self.utilities["WAR1"]:
            game_tree_dict["11"].set_utility_player1( other_agent.utilities["CAP2"] )
            game_tree_dict["11"].set_utility_player2( self.utilities["CAP2"])
            game_tree = self.erase_arc("11", "WAR1", game_tree)
            game_tree_dict["11"].set_children([game_tree_dict["CAP2"]])
        else: 
            game_tree_dict["11"].set_utility_player1( other_agent.utilities["WAR1"] )
            game_tree_dict["11"].set_utility_player2( self.utilities["WAR1"])
            game_tree = self.erase_arc("11", "CAP2", game_tree)
            game_tree_dict["11"].set_children([game_tree_dict["WAR1"]])
            
        # NODE 7 PLAYER 1: NEG VS NODE 11
        if other_agent.utilities["NEG"] > game_tree_dict["11"].get_utility()[PLAYER_1]:
            game_tree_dict["7"].set_utility_player1( other_agent.utilities["NEG"] )
            game_tree_dict["7"].set_utility_player2( self.utilities["NEG"] )
            game_tree = self.erase_arc("7", "11", game_tree)
            game_tree_dict["7"].set_children([game_tree_dict["NEG"]])
        else:
            game_tree_dict["7"].set_utility_player1( game_tree_dict["11"].get_utility()[PLAYER_1])
            game_tree_dict["7"].set_utility_player2( game_tree_dict["11"].get_utility()[PLAYER_2])
            game_tree = self.erase_arc("7", "NEG", game_tree)
            game_tree_dict["7"].set_children([game_tree_dict["11"]])
        
        # NODE 8 
        if other_agent.utilities["CAP1"] > other_agent.utilities["WAR2"]:
            game_tree_dict["8"].set_utility_player1( other_agent.utilities["CAP1"] )
            game_tree_dict["8"].set_utility_player2( self.utilities["CAP1"] )
            game_tree = self.erase_arc("8", "WAR2", game_tree)
            game_tree_dict["8"].set_children([game_tree_dict["CAP1"]])
        else:
            game_tree_dict["8"].set_utility_player1( other_agent.utilities["WAR2"] )
            game_tree_dict["8"].set_utility_player2( self.utilities["WAR2"] )
            game_tree = self.erase_arc("8", "CAP1", game_tree)
            game_tree_dict["8"].set_children([game_tree_dict["WAR2"]])
        
        # NODE 6 PLAYER 1 IN THER PERSPECTIVE OF PLAYER 2
        if game_tree_dict["7"].get_utility()[PLAYER_2] > game_tree_dict["8"].get_utility()[PLAYER_2]:
            game_tree_dict["6"].set_utility_player1( game_tree_dict["7"].get_utility()[PLAYER_1] )
            game_tree_dict["6"].set_utility_player2( game_tree_dict["7"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("6", "8", game_tree)
            game_tree_dict["6"].set_children([game_tree_dict["7"]])
        else:
            game_tree_dict["6"].set_utility_player1( game_tree_dict["8"].get_utility()[PLAYER_1] )
            game_tree_dict["6"].set_utility_player2( game_tree_dict["8"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("6", "7", game_tree)
            game_tree_dict["6"].set_children([game_tree_dict["8"]])
        
        # NODE 4
        if other_agent.utilities["ACQ1"] > game_tree_dict["6"].get_utility()[PLAYER_1]:
            game_tree_dict["4"].set_utility_player1( other_agent.utilities["ACQ1"] )
            game_tree_dict["4"].set_utility_player2( self.utilities["ACQ1"] )
            game_tree = self.erase_arc("4", "6", game_tree)
            game_tree_dict["4"].set_children([game_tree_dict["ACQ1"]])
        else:
            game_tree_dict["4"].set_utility_player1( game_tree_dict["6"].get_utility()[PLAYER_1] )
            game_tree_dict["4"].set_utility_player2( game_tree_dict["6"].get_utility()[PLAYER_2])
            game_tree = self.erase_arc("6", "ACQ1", game_tree)
            game_tree_dict["4"].set_children([game_tree_dict["6"]])
        
        # NODE 2
        if self.utilities["SQ"] > game_tree_dict["4"].get_utility()[PLAYER_2]:
            game_tree_dict["2"].set_utility_player1( other_agent.utilities["SQ"] )
            game_tree_dict["2"].set_utility_player2( self.utilities["SQ"] )
            game_tree = self.erase_arc("2", "4", game_tree)
            game_tree_dict["2"].set_children([game_tree_dict["SQ"]])
        else:
            game_tree_dict["2"].set_utility_player1( game_tree_dict["4"].get_utility()[PLAYER_1] )
            game_tree_dict["2"].set_utility_player2( game_tree_dict["4"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("2", "SQ", game_tree)
            game_tree_dict["2"].set_children([game_tree_dict["4"]])
            
        # NODE 1
        if game_tree_dict["2"].get_utility()[PLAYER_1] > game_tree_dict["3"].get_utility()[PLAYER_1]:
            game_tree_dict["1"].set_utility_player1( game_tree_dict["2"].get_utility()[PLAYER_1] )
            game_tree_dict["1"].set_utility_player2( game_tree_dict["2"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("1", "3", game_tree)
            game_tree_dict["1"].set_children([game_tree_dict["2"]])
        else:
            game_tree_dict["1"].set_utility_player1( game_tree_dict["3"].get_utility()[PLAYER_1] )
            game_tree_dict["1"].set_utility_player2( game_tree_dict["3"].get_utility()[PLAYER_2] )
            game_tree = self.erase_arc("1", "2", game_tree)
            game_tree_dict["1"].set_children([game_tree_dict["3"]])

        return game_tree, game_tree_dict
        
    # given the current node where the agent is in the game,
    # this function will check which actions are available for the other agent to take
    # and predicts what will be the best utility that agent1 can get from the other agent's actions
    def backward_induction(self, other_agent, game_tree, game_tree_dict):

        return [ game_tree, game_tree_dict if  self.perspective_player1(other_agent, game_tree, game_tree_dict) 
                                          else self.perspective_player2(other_agent, game_tree, game_tree_dict)]
    
    
    def apply_action(self, other_agent, game_tree_dict):
        
        # Choose the action with the highest utility
        current_node = self.get_current_node()

        self.set_current_action( game_tree_dict[str(current_node)].get_action() )
        
        # add action to the records
        self.history.append(self.get_current_action())
        
        # Perform the action
        print(f"\t{self.name} used {self.current_action}")
        
    def update_state(self, other_agent):
        
        print("UPDATE STATE")
        
        print(self.current_node)
        print(self.current_action)
        
        if (self.current_action == "apply force") and (self.current_node == 1):
            other_agent.current_node = 3 # move to node 3
            other_agent.start_node = 3 # move to node 3

            return
        if (self.current_action == "do nothing") and (self.current_node == 1):
            other_agent.current_node = 2 # move to node 10
            other_agent.start_node = 2 # move to node 10
            return
        
        if (self.current_action == "apply force") and (self.current_node == 2):
            other_agent.current_node = 4 # move to node 10
            return
        if (self.current_action == "do nothing") and (self.current_node == 2):
            other_agent.game_ended = "SQ"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 3):
            other_agent.current_node = 5 # move to node 5
            return
        if (self.current_action == "do nothing") and (self.current_node == 3):
            other_agent.game_ended = "ACQ2"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 4):
            other_agent.current_node = 6 # move to node 6
            return
        if (self.current_action == "do nothing") and (self.current_node == 4):
            other_agent.game_ended = "ACQ1"
            return
    
        if (self.current_action == "apply force") and (self.current_node == 5):
            other_agent.current_node = 10 # move to node 10
            #other_agent.start_node = 10 # move to node 10
            return 
        if (self.current_action == "do nothing") and (self.current_node == 5):
            other_agent.current_node = 9 # move to node 9
            #other_agent.start_node = 9 # move to node 10
            return
        
        if (self.current_action == "apply force") and (self.current_node == 6):
            other_agent.current_node = 8 # move to node 10
            return 
        if (self.current_action == "do nothing") and (self.current_node == 6):
            other_agent.current_node = 7 # move to node 9
            return
        
        if (self.current_action == "apply force") and (self.current_node == 7):
            other_agent.current_node = 10 
            return 
        if (self.current_action == "do nothing") and (self.current_node == 7):
            other_agent.game_ended = "WAR1"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 8):
            self.game_ended = "WAR2"
            return
        if (self.current_action == "do nothing") and (self.current_node == 8):
            self.game_ended = "CAP1"
            return
        
        if (self.current_node == 9) and (self.current_action == "apply force"):
            # switch branch and go to node 8
            other_agent.current_node = 8
            return 
        if (self.current_node == 9) and (self.current_action == "do nothing"):
            # switch branch and go to node 8
            self.game_ended = "NEG"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 10):
            other_agent.game_ended = "WAR1"
            return
        if (self.current_action == "do nothing") and (self.current_node == 10):
            other_agent.game_ended = "CAP2"
            return
        