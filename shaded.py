# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:16:14 2023

@author: zhanghai
"""

import time
import copy
import numpy as np
import pandas as pd
from math import factorial
from scipy.special import comb
from itertools import combinations
from pyds import MassFunction



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




def detect_deceptive(mass_list, n_states):
    # Omega = [chr(ord('a') + i) for i in range(n_states)]
    n_mass = len(mass_list)
    reasonable_fusion = weighted_average_mass(mass_list, n_states).pignistic()
    reasonable_p = list(reasonable_fusion.values())
    reasonable_t = list(reasonable_fusion.keys())
    reasonable_target = list(reasonable_t[reasonable_p.index(max(reasonable_p))])[0]
    print("Intutive target: ", reasonable_target)
    rule_fusion = mass_list[0]
    for i in range(1, n_mass):
        rule_fusion = rule_fusion&mass_list[i]
    rule_fusion = rule_fusion.pignistic()
    rule_p = list(rule_fusion.values())
    rule_t = list(rule_fusion.keys())
    rule_target = list(rule_t[rule_p.index(max(rule_p))])[0]
    print("Combined target: ", rule_target)    
    
    # calculate shapley values
    m0 = MassFunction({'abc':1})
    mass_list.insert(0, m0)
    stack = [frozenset([0])]
    combined_mass = {frozenset([0]): m0}
    p_target = {frozenset([0]): 1/n_states}
    predicted_target = {frozenset([0]): reasonable_target}

    while len(stack)!=0:
        parent_index = stack.pop()
        parent_mass = combined_mass[parent_index]
        level = max(list(parent_index))
        # print(stack)
        for j in range(n_mass,level,-1):
            current_index = parent_index | frozenset([j])
            combined_mass[current_index] = parent_mass.combine_conjunctive(mass_list[j])
            current_fusion = combined_mass[current_index].pignistic()
            current_p = list(current_fusion.values())
            current_t = list(current_fusion.keys())
            current_target = list(current_t[current_p.index(max(current_p))])[0]
            predicted_target[current_index] = current_target
            p_target[current_index] = current_fusion[reasonable_target]
            stack.append(current_index)

    # weights for each subset in shapley value    
    weights = np.zeros((n_mass, 2**(n_mass-1)))
    current_index = 0
    for s in range(n_mass):
        n_comb = int(comb(n_mass-1, s))
        current_weight = factorial(s)*factorial(n_mass-s-1)/factorial(n_mass)
        weights[:, current_index:(current_index+n_comb)] = current_weight
        current_index += n_comb

    # calculation of shapley values    
    differences = np.zeros((n_mass, 2**(n_mass-1)))
    for i in range(1, n_mass+1):
        elements_except_i = list(range(1,n_mass+1))
        elements_except_i.remove(i)
        index_with_i = []
        index_except_i = []
        for s in range(len(elements_except_i)+1):
            for subset in combinations(elements_except_i, s):
                index_except_i.append(frozenset([0]) | frozenset(subset))
                index_with_i.append(frozenset([0,i]) | frozenset(subset))
                
        for s in range(len(index_except_i)):
            differences[i-1,s] = p_target[index_with_i[s]] - p_target[index_except_i[s]]
            
    shapley_values = (differences * weights).sum(axis=1)

    # get the set of credible mass funcions
    CE = np.where(shapley_values>=0)[0] + 1
        
    string = "Credible evidence are "+''.join('m'+str(e)+', ' for e in CE[1:])
    string = string[:-2] + '.'
    print(string)
    
    
    DE = list(set(range(1,n_mass+1)).difference(set(CE)))
    
    if rule_target == reasonable_target:
        WDE = DE
        strong_deceptive = []
        string = "No strong deceptive evidence. \nWeak deceptives are "+''.join('m'+str(e)+', ' for e in WDE)
        string = string[:-2] + '.'
        print(string)
    else:
        SDE = []
        i = 1
        while i <= len(DE):
            tmp = set()
            for subset in combinations(DE, i):
                
                index = frozenset({0})|frozenset(CE)|frozenset(subset)
                if predicted_target[index] != reasonable_target:
                    SDE.append(subset)
                    tmp = tmp.union(set(subset))
            DE = list(set(DE).difference(tmp))
            i += 1
            
        WDE = DE
        
        if len(WDE) > 0:
            string = "Weak deceptives are "+''.join('m'+str(e)+', ' for e in WDE)
            string = string[:-2] + '.'
            print(string)
        else:
            print("No weak deceptive evidence.")
            
            
        string = "Strong deceptives are "
        for e in SDE:
            if len(e) == 1:
                string += "m{}, ".format(e[0])
            else:
                string += '('+''.join('m'+str(i)+', ' for i in e)
                string = string[:-2] + '), '
        string = string[:-2] + '.'
        print(string)
        
    
    return shapley_values, SDE, DE, p_target, predicted_target
    




if __name__ == '__main__':
    
    # test example 1
    # m1 = MassFunction({'a':0.8, 'b':0.2, 'c':0})
    # m2 = MassFunction({'a':0.6, 'b':0.1, 'c':0.3})
    # m3 = MassFunction({'a':0.2, 'b':0.35, 'c':0.45}) # m3 is deceptive
    # mass_list = [m1, m2,m3]
    
    # test example 2.1
    # m1 = MassFunction({'a':0.7, 'b':0.1, 'c':0, 'abc':0.2})
    # m2 = MassFunction({'a':0.65, 'b':0.15, 'c':0, 'abc':0.2}) 
    # m3 = MassFunction({'a':0.75, 'b':0, 'c':0.05, 'abc':0.2})
    # m4 = MassFunction({'a':0.1, 'b':0.1, 'c':0.8}) # m5 is deceptive
    # m5 = MassFunction({'a':0, 'b':0.9, 'c':0.1}) # m6 is deceptive
    # mass_list = [m1, m2, m3, m4, m5] 
    
    # test example 2.2
    # m1 = MassFunction({'a':0.41, 'b':0.29, 'c':0.3})
    # m2 = MassFunction({'a':0, 'b':0.9, 'c':0.1})
    # m3 = MassFunction({'a':0.58, 'b':0.07, 'c':0, 'ac':0.35})
    # m4 = MassFunction({'a':0.55, 'b':0.1, 'c':0, 'ac':0.35})
    # m5 = MassFunction({'a':0.6, 'b':0.1, 'c':0, 'ac':0.3})
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
    
    n_states = 3
    start_time = time.time()
    shapley_values, strong_deceptive, weak_deceptive, p, t = detect_deceptive(mass_list, n_states)
    end_time = time.time()
    print('Shapley values are ', shapley_values)
    print('time cost is {}s'.format(round(end_time-start_time, 4)))