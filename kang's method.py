# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:52:40 2024

@author: zhanghai
"""


import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyds import MassFunction
from itertools import combinations




def deng_entropy(mass):
    entrop = 0
    focal_elements = list(mass.keys())
    
    for focal_element in focal_elements:
        if mass[focal_element] == 0:
            continue
        cardinality = len(focal_element)
        entrop -= mass[focal_element]*np.log2(mass[focal_element]/(2**cardinality-1))
        
    return entrop

def paired_evidence_distance(m1, m2, n_states_nature):
    if n_states_nature > 26:
        print('number of states too large!')
        return None

    all_states = [chr(ord('a') + i) for i in range(n_states_nature)]
    cardinality = 2**n_states_nature - 1
    all_subsets = []
    
    for i in range(1, n_states_nature+1):
        for subset in combinations(all_states, i):
            all_subsets.append(frozenset(subset))

    m1_ = pd.Series(np.zeros(cardinality), all_subsets)
    m2_ = pd.Series(np.zeros(cardinality), all_subsets)
    for focal_element in list(m1.keys()):
        m1_[focal_element] = m1[focal_element]
    for focal_element in list(m2.keys()):
        m2_[focal_element] = m2[focal_element]
    mass_difference = m1_ - m2_
    
    D_matrix = pd.DataFrame(np.ones((cardinality, cardinality)), 
                            index=all_subsets,
                            columns=all_subsets)
    
    for i in range(cardinality):
        for j in range(i, cardinality):
            row_index = all_subsets[i]
            col_index = all_subsets[j]
            d_ij = len(row_index & col_index)/len(row_index | col_index)
            D_matrix.iloc[i, j] = d_ij
    upper_indices = np.triu_indices(cardinality, k=1)
    D_matrix.values.T[upper_indices] = D_matrix.values[upper_indices]
    
    distance = np.sqrt(0.5 * mass_difference.values.T@D_matrix.values@mass_difference.values)
    return distance


def weighted_average_mass(mass_list, n_states_nature, weights=None):
    if len(mass_list) == 1:
        return mass_list[0]
    n_mass = len(mass_list)
    if weights is None:
        distance_matrix = np.zeros((n_mass, n_mass))
        for i in range(n_mass):
            for j in range(i+1, n_mass):
                distance_matrix[i,j] = paired_evidence_distance(mass_list[i], 
                                                                mass_list[j], 
                                                                n_states_nature)
        upper_indices = np.triu_indices(n_mass, k=1)
        distance_matrix.T[upper_indices] = distance_matrix[upper_indices]
        similarity_matrix = 1 - distance_matrix
        weights = similarity_matrix.sum(axis=1) - 1
        weights = weights/sum(weights)
    
    # averaging discounted mass functions
    # Omega = frozenset([chr(ord('a') + i) for i in range(n_states_nature)])
    # all_focal_elements = [Omega]
    # for i in range(n_mass):
    #     all_focal_elements = set(all_focal_elements).union(set(list(weighted_mass_list[i].keys())))
    #     # print(weighted_mass_list[i])
    #     for focal_element in list(weighted_mass_list[i].keys()):
    #         weighted_mass_list[i][focal_element] *= weights[i]
            
    #     if Omega in weighted_mass_list[i].keys():
    #         weighted_mass_list[i][Omega] += (1 - weights[i])
    #     else:
    #         weighted_mass_list[i][Omega] = (1 - weights[i])
    
    # averaged_mass = MassFunction()
    # for focal_element in all_focal_elements:
    #     averaged_mass[focal_element] = 0
    #     for i in range(n_mass):
    #         averaged_mass[focal_element] += weighted_mass_list[i][focal_element]/n_mass
    
    
    # weignted averaging original mass functions
    all_focal_elements = []
    for i in range(n_mass):
        all_focal_elements = set(all_focal_elements).union(set(list(mass_list[i].keys())))
    averaged_mass = MassFunction()
    for focal_element in all_focal_elements:
        averaged_mass[focal_element] = 0
        for i in range(n_mass):
            averaged_mass[focal_element] += mass_list[i][focal_element]*weights[i]
    
    
    aggregated_mass = copy.deepcopy(averaged_mass)
    for i in range(n_mass-1):
        aggregated_mass = aggregated_mass&averaged_mass
        
    return aggregated_mass


def reward_condistions(mass_list, state, n_states_nature):
    # calculate entropy for the next state
    included_mass = list(state)
    m = mass_list[included_mass[0]]
    if len(included_mass) > 1:
        for i in range(1, len(included_mass)):
            m = m&mass_list[included_mass[i]]
            
    entropy = deng_entropy(m)
    
    pignistic_proabas = m.pignistic()
    max_p = 0
    for key in pignistic_proabas.keys():
        if pignistic_proabas[key] > max_p:
            max_p = pignistic_proabas[key]
            predicted_target_DCR = list(key)[0]

    state_mass_list = []
    for i in range(len(included_mass)):
        state_mass_list.append(mass_list[included_mass[i]])
    m = weighted_average_mass(state_mass_list, n_states_nature)
    pignistic_proabas = m.pignistic()
    max_p = 0
    for key in pignistic_proabas.keys():
        if pignistic_proabas[key] > max_p:
            max_p = pignistic_proabas[key]
            predicted_target_WA = list(key)[0]
            
    return entropy, predicted_target_DCR, predicted_target_WA
    
    

def Q_learning(mass_list, n_states_nature, n_episodes=5000, discounting_factor=0.1, learning_rate=0.2, exploration_rate=0.1):
    # initialize Q_table
    n_mass = len(mass_list)
    current_state = frozenset([])
    row_values = np.zeros(n_mass)
    all_actions = np.arange(n_mass)
    Q_table = {current_state:row_values}
    entropy_list =  {current_state:-np.log2(1/(2**n_states_nature-1))}
    predicted_target_DCR = {current_state:'a'}
    predicted_target_WA = {current_state:'a'}
    
    global_fusion = weighted_average_mass(mass_list, n_states_nature).pignistic()
    global_p = list(global_fusion.values())
    global_t = list(global_fusion.keys())
    global_target = list(global_t[global_p.index(max(global_p))])[0]
    
    for episode in tqdm(range(n_episodes)):
        current_state = frozenset([])
            
        while True:
            if len(current_state) == 0:
                action = np.random.choice(n_mass)
            else:
                available_actions = np.setdiff1d(all_actions, list(current_state))
                pi_a_s = 1 - exploration_rate
                if np.random.rand() < pi_a_s:
                    max_indices = np.where(Q_table[current_state] == np.max(Q_table[current_state]))[0]
                    action = np.random.choice(max_indices)
                    # action = Q_table[current_state].argmax()
                else:
                    action = np.random.choice(available_actions)
            
            next_state = current_state | frozenset([action])
            # if len(list(next_state)) == len(mass_list):
            #     break
            
            # print(current_state, next_state)
            if next_state not in Q_table.keys():
                # insert next_state to Q_table with intial row value
                row_values = np.zeros(n_mass)
                row_values[list(next_state)] = -100
                Q_table[next_state] = row_values
                
                e, t1, t2 = reward_condistions(mass_list, next_state, n_states_nature)
                entropy_list[next_state] = e
                predicted_target_DCR[next_state] = t1
                predicted_target_WA[next_state] = t2
            else:
                e = entropy_list[next_state]
                t1 = predicted_target_DCR[next_state]
                t2 = predicted_target_WA[next_state]
            
            # if (e <= entropy_list[current_state]):
                # if (t1==t2):
                #     reward = 10
                # elif (t1==global_target):
                #     reward = 15
                # else:
                #     reward = -10
            if (e <= entropy_list[current_state]) and (t1==global_target):
                reward = 10
            else:
                reward = -10
                # next_state = current_state
                
            # update Q-table
            current_q = Q_table[current_state][action]
            if len(next_state) == n_mass:
                max_next_q = 0
            else:
                max_next_q = Q_table[next_state].max()
            Q_table[current_state][action] = current_q + learning_rate*(reward + discounting_factor*max_next_q - current_q)
            current_state = next_state
            if len(list(current_state)) == n_mass:
                break
            
    return Q_table, entropy_list, predicted_target_DCR, predicted_target_WA
       
 
def select_reliable_evidence(mass_list, n_states_nature, Q_table, entropy_list, predicted_target_DCR, predicted_target_WA):
    current_state = frozenset([])
    while True:
        if len(list(current_state)) == len(mass_list):
            break
        # print(current_state, Q_table[current_state])
        action = Q_table[current_state].argmax()
        next_state = current_state | frozenset([action])
        
        # calculate entropy for the next state
        e, t1, t2 = reward_condistions(mass_list, next_state, n_states_nature)
        entropy_list[next_state] = e
        predicted_target_DCR[next_state] = t1
        predicted_target_WA[next_state] = t2
    
        if (entropy_list[next_state] <= entropy_list[current_state]) and (predicted_target_DCR[next_state]==predicted_target_WA[next_state]):
            current_state = next_state
        else:
            break
    
    return np.array(list(current_state))+1





if __name__ == '__main__':
    # test example 1
    # m1 = MassFunction({'a':0.8, 'b':0.2, 'c':0})
    # m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0.3})
    # m3 = MassFunction({'a':0.2, 'b':0.35, 'c':0.45}) # m3 is deceptive
    # mass_list = [m1, m2,m3]
    
    # test example 2
    # m1 = MassFunction({'a':0.7, 'b':0.1, 'c':0, 'abc':0.2})
    # m2 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
    # m3 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
    # m4 = MassFunction({'a':0.1, 'b':0.1, 'c':0.8}) # m5 is deceptive
    # m5 = MassFunction({'a':0, 'b':0.9, 'c':0.1}) # m6 is deceptive
    # mass_list = [m1, m2, m3, m4, m5] 
    
    # test example 3
    m1 = MassFunction({'a':0.3, 'b':0.6, 'c':0, 'abc':0.1}) # m1 is deceptive
    m2 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
    m3 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
    m4 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
    m5 = MassFunction({'a':0.05, 'b':0.45, 'c':0.5, 'abc':0}) # m5 is deceptive
    m6 = MassFunction({'a':0.05, 'b':0.5, 'c':0.45, 'abc':0}) # m6 is deceptive
    mass_list = [m1, m2, m3, m4, m5, m6] 
    
    # test example 4
    # m1 = MassFunction({'a':0.0, 'b':0.8, 'c':0.2}) # m1 is deceptive
    # m2 = MassFunction({'a':0.0, 'b':0.2, 'c':0.8}) # m2 is deceptive
    # m3 = MassFunction({'a':0.5, 'b':0.4, 'c':0, 'abc':0.1})
    # m4 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
    # m5 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
    # m6 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
    # m7 = MassFunction({'a':0.65, 'b':0.1, 'c':0, 'ac':0.25})
    # m8 = MassFunction({'a':0.6, 'b':0.2, 'c':0, 'ac':0.2})
    # m9 = MassFunction({'a':0.5, 'b':0.2, 'c':0, 'ac':0.3})
    # m10 = MassFunction({'abc':1})
    # m11 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
    # m12 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
    # m13 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
    # m14 = MassFunction({'a':0.65, 'b':0.1, 'c':0, 'ac':0.25})
    # m15 = MassFunction({'a':0.6, 'b':0.2, 'c':0, 'ac':0.2})
    # m16 = MassFunction({'a':0.5, 'b':0.2, 'c':0, 'ac':0.3})
    # m17 = MassFunction({'a':0.6, 'b':0.2, 'c':0, 'ac':0.2})
    # m18 = MassFunction({'a':0.5, 'b':0.2, 'c':0, 'ac':0.3})
    # m19 = MassFunction({'abc':1})
    # m20 = MassFunction({'a':0.7, 'b':0, 'c':0, 'abc':0.3})
    
    # mass_list = [m1, m2, m3, m4, m5] 
    
    # mass_list = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10] 
    
    # mass_list = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
    #               m11, m12, m13, m14, m15] 
    
    # mass_list = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
    #               m11, m12, m13, m14, m15, m16, m17, m18, m19, m20] 
    
    
    n_states_nature = 3
    start_time = time.time()
    a,b,c,d = Q_learning(mass_list, n_states_nature)
    end_time = time.time()
    crediable_evidence = select_reliable_evidence(mass_list, n_states_nature, a, b, c, d)
    deceptive_evidence = set(range(1, len(mass_list)+1)).difference(set(crediable_evidence))
    print("deceptive evidences are "+''.join('m'+str(e)+', ' for e in deceptive_evidence))
    print('time cost is ', round(end_time-start_time, 4))
