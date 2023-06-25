import random
import numpy as np
import pandas as pd


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Agent:

    def __init__(self, name, id, utilities):
        self.name = name                # name of the country
        self.id = id                    # checks whether the agent is the first player (id = 1) or the second player (id = 2)
        self.actions = [                # the actions that the agent can take
            "apply force",      
            "do nothing"
        ]
        self.current_action = None      # the action that the agent is currently taking
        self.history = []               # keeping track of the actions taken by the agent [for debugging purposes]
        self.current_node = -1          # keeping track of where the agent is in the tree
        self.start_node = -1            # the node where the agent starts
        self.utilities = utilities
        self.game_ended = "No"
        
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
    
    def backward_induction(self, other_agent):
        
        if self.id == 2: # game from p2 perspective
            
            util_node_12_p1 = np.max( [other_agent.utilities["CAP1"], other_agent.utilities["WAR2"]] )
            util_node_11_p1 = [ other_agent.utilities["CAP2"] if self.utilities["CAP2"] > self.utilities["WAR1"] else other_agent.utilities["WAR1"]  ]
            util_node_10_p1 = [ other_agent.utilities["CAP2"] if self.utilities["CAP2"] > self.utilities["WAR1"] else other_agent.utilities["WAR1"] ]
            util_node_9_p1 = [  other_agent.utilities["NEG"] if self.utilities["NEG"] > util_node_12_p2 else util_node_12_p1 ]
            util_node_8_p1 = np.max([other_agent.utilities["CAP1"], other_agent.utilities["WAR2"]])
            util_node_7_p1 = np.max([util_node_11_p1, other_agent.utilities["NEG"]])
            util_node_6_p1 = [  util_node_7_p1 if util_node_7_p2 > util_node_8_p2 else util_node_8_p1  ]
            util_node_5_p1 = np.max( [util_node_9_p1, util_node_10_p1] )
            util_node_4_p1 = np.max( [util_node_6_p1, self.utilities["ACQ1"]] )
            util_node_3_p1 = [ other_agent.utilities["ACQ2"] if self.utilities["ACQ2"] > util_node_5_p2 else util_node_5_p1   ]
            util_node_2_p1 = [ util_node_4_p1 if other_agent.utilities["SQ"] > self.utilities["SQ"] else other_agent.utilities["SQ"] ]    # player 2
            
            util_node_12_p2 = [ self.utilities["CAP1"] if other_agent.utilities["CAP1"] > other_agent.utilities["WAR2"] else self.utilities["WAR2"]]
            util_node_11_p2 = np.max([self.utilities["CAP2"], self.utilities["WAR1"]])
            util_node_10_p2 = np.max([self.utilities["CAP2"], self.utilities["WAR1"]])
            util_node_9_p2 = np.max([self.utilities["NEG"], util_node_12_p2])
            util_node_8_p2 = [ self.utilities["CAP1"] if other_agent.utilities["CAP1"] > other_agent.utilities["WAR2"] else self.utilities["WAR2"]]
            util_node_7_p2 = [ self.utilities["NEG"] if other_agent.utilities["NEG"] > util_node_11_p1 else util_node_11_p2]
            util_node_6_p2 = np.max( [util_node_7_p2, util_node_8_p2])
            util_node_5_p2 = [ util_node_9_p2 if util_node_9_p1 > util_node_10_p1 else util_node_10_p2 ]
            util_node_4_p2 = [ util_node_6_p2 if util_node_6_p1 > other_agent.utilities["ACQ1"] else self.utilities["ACQ1"]]
            util_node_3_p2 = np.max( util_node_5_p2, self.utilities["ACQ2"])
            util_node_2_p2 = np.max( util_node_4_p2, self.utilities["SQ"]  )
            util_node_1_p2 = [ util_node_2_p2 if util_node_2_p1 > util_node_3_p1 else util_node_3_p2 ]
            return [util_node_1_p2 ]
            
        
        # if it is player 1's move, do the following
        if self.id == 1: # game from p1 perspective
            # player 1's moves
            util_node_12_p2 = [ other_agent.utilities["CAP1"] if self.utilities["CAP1"] > self.utilities["WAR2"] else other_agent.utilities["WAR2"]]
            util_node_11_p2 = np.max([other_agent.utilities["CAP2"], other_agent.utilities["WAR1"]])
            util_node_10_p2 = np.max([other_agent.utilities["CAP2"], other_agent.utilities["WAR1"]])
            util_node_9_p2 = np.max([other_agent.utilities["NEG"], util_node_12_p2])
            util_node_8_p2 = [ other_agent.utilities["CAP1"] if self.utilities["CAP1"] > self.utilities["WAR2"] else other_agent.utilities["WAR2"]]
            util_node_7_p2 = [ other_agent.utilities["NEG"] if self.utilities["NEG"] > util_node_11_p1 else util_node_11_p2]
            util_node_6_p2 = np.max( [util_node_7_p2, util_node_8_p2])
            util_node_5_p2 = [ util_node_9_p2 if util_node_9_p1 > util_node_10_p1 else util_node_10_p2 ]
            util_node_4_p2 = [ util_node_6_p2 if util_node_6_p1 > self.utilities["ACQ1"] else other_agent.utilities["ACQ1"]]
            util_node_3_p2 = np.max( util_node_5_p2, other_agent.utilities["ACQ2"])
            util_node_2_p2 = np.max( util_node_4_p2, other_agent.utilities["SQ"]  )
            util_node_1_p2 = [ util_node_2_p2 if util_node_2_p1 > util_node_3_p1 else util_node_3_p2 ]
            
            
            util_node_12_p1 = np.max( [self.utilities["CAP1"], self.utilities["WAR2"]] )
            util_node_11_p1 = [ self.utilities["CAP2"] if other_agent.utilities["CAP2"] > other_agent.utilities["WAR1"] else self.utilities["WAR1"]  ]
            util_node_10_p1 = [ self.utilities["CAP2"] if other_agent.utilities["CAP2"] > other_agent.utilities["WAR1"] else self.utilities["WAR1"] ]
            util_node_9_p1 = [  self.utilities["NEG"] if other_agent.utilities["NEG"] > util_node_12_p2 else util_node_12_p1 ]
            util_node_8_p1 = np.max([self.utilities["CAP1"], self.utilities["WAR2"]])
            util_node_7_p1 = np.max([util_node_11_p1, self.utilities["NEG"]])
            util_node_6_p1 = [  util_node_7_p1 if util_node_7_p2 > util_node_8_p2 else util_node_8_p1  ]
            util_node_5_p1 = np.max( [util_node_9_p1, util_node_10_p1] )
            util_node_4_p1 = np.max( [util_node_6_p1, self.utilities["ACQ1"]] )
            util_node_3_p1 = [ self.utilities["ACQ2"] if other_agent.utilities["ACQ2"] > util_node_5_p2 else util_node_5_p1   ]
            util_node_2_p1 = [ util_node_4_p1 if self.utilities["SQ"] > other_agent.utilities["SQ"] else self.utilities["SQ"] ]    # player 2
            util_node_1_p1 = max( util_node_2_p1, util_node_3_p1 )  # player 1
            return util_node_1_p1
        
    
    # given the current node where the agent is in the game,
    # this function will check which actions are available for the other agent to take
    # and predicts what will be the best utility that agent1 can get from the other agent's actions
    def forward_checking(self, other_agent): 
        
        util_action_force = 0
        util_action_nothing = 0
        
        # NODE 5 ###########################################################################################
        if self.current_node == 5:

            # if the agent chooses to apply force, then this will move the game to node 10,
            # this means that if agent2 reacts with apply force, then the end state will be WAR1
            # otherwise, if the other agent reacts with do nothing, then the end state will be CAP2
            util_action_force = max(other_agent.utilities["WAR1"], other_agent.utilities["CAP2"])
                
            # if the agent chooses to do nothing, then this will move the game to node 9,    
            # if the other agent reacts with do nothing, then the end state will be NEG
            # if the other agent reacts with force, then the game will move to node 8
            # and then it is up to the current agent to decide whether to apply force or do nothing
            util_action_nothing = max( other_agent.utilities["NEG"], self.utilities["WAR2"], self.utilities["CAP1"])
            
            # return the final preferred action for node 5 and end forward checking
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 9 ###########################################################################################
        if self.current_node == 9:
                
            # if the other agents reacts with do nothing, then the end state will be NEG
            util_action_nothing = self.utilities["NEG"]
            
            # if the other agent reacts with force, then the game will move to node 8
            # and then it is up to the current agent to decide whether to apply force or do nothing
            util_action_force = max(other_agent.utilities["WAR2"], other_agent.utilities["CAP1"])

            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 10 ###########################################################################################
        if self.current_node == 10:
            
            # if the other agent applies force, then the end state will be WAR1, 
            # otherwise, if the other agent does nothing, then the end state will be CAP2
            util_action_nothing = self.utilities["CAP2"]
            util_action_force = self.utilities["WAR1"]
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 6 ###############################################################################################
        if self.current_node == 6:
            
            # if the agent applies force, then if the other agent applies force, the end state will be WAR2, 
            # otherwise, if the other agent does nothing, then the end state will be CAP1
            util_action_force = max(other_agent.utilities["WAR2"], other_agent.utilities["CAP1"])
            
            # if the agent does nothing, if the other agent also does nothing, the end state will be NEG
            # otherwise, if the other agent applies force, the game will jump to node 7
            util_action_nothing = other_agent.utilities["NEG"]
            util_action_force = max(util_action_force, self.utilities["CAP2"], self.utilities["WAR1"])
        
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 8 ###############################################################################################
        if self.current_node == 8:
            
            util_action_force = self.utilities["WAR2"]
            util_action_nothing = self.utilities["CAP1"]
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action

        # NODE 7 ###############################################################################################
        if self.current_node == 7:
            
            util_action_nothing = self.utilities["NEG"]
            util_action_force = max(other_agent.utilities["CAP2"], other_agent.utilities["WAR1"])
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 3 ###############################################################################################
        if self.current_node == 3:
            
            util_action_nothing = self.utilities["ACQ2"]
            util_action_force =  max(self.utilities["WAR1"], self.utilities["CAP2"], self.utilities["NEG"], other_agent.utilities["CAP1"], other_agent.utilities["WAR2"])
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 4 ###############################################################################################
        if self.current_node == 4:
            
            util_action_nothing = self.utilities["ACQ1"]
            util_action_force =  max(self.utilities["CAP1"], self.utilities["WAR2"], self.utilities["NEG"], other_agent.utilities["CAP2"], other_agent.utilities["WAR1"])
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 2 ###############################################################################################
        if self.current_node == 2:
            util_action_nothing = self.utilities["SQ"]
            util_action_force =  max(other_agent.utilities["ACQ1"], other_agent.utilities["CAP1"], other_agent.utilities["WAR2"], other_agent.utilities["NEG"], self.utilities["CAP2"], self.utilities["WAR1"])
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
        # NODE 1 ###############################################################################################
        if self.current_node == 1:
            
            util_action_nothing = max(self.utilities["ACQ1"], other_agent.utilities["SQ"], self.utilities["NEG"], self.utilities["CAP1"], self.utilities["WAR2"], other_agent.utilities["CAP2"], other_agent.utilities["WAR1"])
            util_action_force =  max(self.utilities["WAR2"], self.utilities["CAP1"], other_agent.utilities["NEG"], other_agent.utilities["CAP2"], other_agent.utilities["WAR1"], other_agent.utilities["ACQ2"])
            
            self.current_action = "apply force" if util_action_force >= util_action_nothing else "do nothing"
            return self.current_action
        
    def apply_action(self, other_agent):
        
        if self.start_node == -1:
            self.current_node = 5
            self.start_node = 5
        
        # Choose the action with the highest utility
        action = self.forward_checking(other_agent)
        
        # add action to the records
        self.history.append(action)
        
        # Perform the action
        print(f"\t{self.name} used {action}")
        
    def update_state(self, other_agent):
        
        print("\tUpdating the state of the world...")
        
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
            other_agent.game_ended = f"   ->SQ to {self.name}"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 3):
            other_agent.current_node = 5 # move to node 5
            return
        if (self.current_action == "do nothing") and (self.current_node == 3):
            other_agent.game_ended = f"   ->ACQ2 to {self.name}"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 4):
            other_agent.current_node = 6 # move to node 6
            return
        if (self.current_action == "do nothing") and (self.current_node == 4):
            other_agent.game_ended = f"   ->ACQ1 to {self.name}"
            return
    
        if (self.current_action == "apply force") and (self.current_node == 5):
            other_agent.current_node = 10 # move to node 10
            #other_agent.start_node = 10 # move to node 10
            return 
        if (self.current_action == "do nothing") and (self.current_node == 5):
            other_agent.current_node = 9 # move to node 9
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
            other_agent.game_ended = f"   ->WAR1 to {self.name}"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 8):
            self.game_ended = f"WAR to {other_agent.name}"
            return
        if (self.current_action == "do nothing") and (self.current_node == 8):
            self.game_ended = f"   ->CAP to {self.name}"
            return
        
        if (self.current_node == 9) and (self.current_action == "apply force"):
            # switch branch and go to node 8
            other_agent.current_node = 8
            return 
        if (self.current_node == 9) and (self.current_action == "do nothing"):
            # switch branch and go to node 8
            self.game_ended = f"   ->Negotiation between {self.name} and {other_agent.name}"
            return
        
        if (self.current_action == "apply force") and (self.current_node == 10):
            other_agent.game_ended = f"   ->WAR1 to {self.name}"
            return
        if (self.current_action == "do nothing") and (self.current_node == 10):
            other_agent.game_ended = f"   ->CAP2 to {self.name}"
            return
        
        
        
import pandas as pd
DATASET_PATH = "/Users/pin083/Documents/GitHub/International_Interaction_Game/dataset/utilities.csv"
utilities = pd.read_csv(DATASET_PATH, sep="\t")

def create_agent_pair( indx : int, utilities ):
    
    util_agent1 = {}
    agent1_name = utilities.loc[indx, "Agent1"]
    util_agent1["SQ"] = utilities.loc[indx, "wrTu1sq"]
    util_agent1["ACQ1"] = utilities.loc[indx, "wrTu1ac1"]
    util_agent1["ACQ2"] = utilities.loc[indx, "wrTu1ac2"]
    util_agent1["NEG"] = utilities.loc[indx, "wrTu1neg"]
    util_agent1["CAP1"] = utilities.loc[indx, "wrTu1cp1"]
    util_agent1["CAP2"] = utilities.loc[indx, "wrTu1cp2"]
    util_agent1["WAR1"] = utilities.loc[indx, "wrTu1wr1"]
    util_agent1["WAR2"] = utilities.loc[indx, "wrTu1wr2"]

    agent2_name = utilities.loc[indx, "Agent2"]
    util_agent2 = {}
    util_agent2["SQ"] = utilities.loc[indx, "wrTu2sq"]
    util_agent2["ACQ1"] = utilities.loc[indx, "wrTu2ac2"]
    util_agent2["ACQ2"] = utilities.loc[indx, "wrTu2ac1"]
    util_agent2["NEG"] = utilities.loc[indx, "wrTu2neg"]
    util_agent2["CAP1"] = utilities.loc[indx, "wrTu2cp2"]
    util_agent2["CAP2"] = utilities.loc[indx, "wrTu2cp1"]
    util_agent2["WAR1"] = utilities.loc[indx, "wrTu2wr2"]
    util_agent2["WAR2"] = utilities.loc[indx, "wrTu2wr1"]

    agent1 = Agent(agent1_name, 1, util_agent1)
    agent2 = Agent(agent2_name, 2, util_agent2)
    
    groundtruth = utilities.loc[indx, "groundtruth"]
    predicted = utilities.loc[indx, "predicted"]

    return agent1, agent2, groundtruth, predicted

outcomes = []
groundtruths = []
for game in range(0, len(utilities)):
    agent1, agent2, groundtruth, predicted = create_agent_pair(game, utilities)
    
    # Let the agents interact
    print(f"GAME {agent1.name} VS {agent2.name}...\n" )
    for i in range(10):
        
        print(f"GAME ITERATION {i}\n")
        
        print(f"AGENT {str(agent1.name)} TURN")
        agent1.apply_action(agent2)
        agent1.update_state(agent2)
        print(f"        DEBUG: node: {str(agent1.current_node), agent1.history, agent1.game_ended}]")
        
        print("\n")
        
        
        print(f"AGENT {str(agent2.name)} TURN")
        agent2.apply_action(agent1)
        agent2.update_state(agent1)
        print(f"        DEBUG: node: {str(agent2.current_node), agent2.history, agent2.game_ended}]")
        
        print("---------------------------------------------------------------------")
        
        if agent1.game_ended != "No":
            print(f"GAME ENDED: {agent1.game_ended}")
            outcomes.append(agent1.game_ended.upper())
            groundtruths.append(groundtruth.upper())
            break
        
        if agent2.game_ended != "No":
            print(f"GAME ENDED: {agent2.game_ended}")
            outcomes.append(agent1.game_ended.upper())
            groundtruths.append(groundtruth.upper())
            break
        
confusion_matrix = confusion_matrix(groundtruths, outcomes)
print(confusion_matrix)

# plot confusion matrix
ax = plt.subplot()
sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap="Blues", fmt="d")

    
