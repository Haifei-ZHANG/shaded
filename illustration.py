# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:40:41 2023

@author: zhanghai
"""

import numpy as np
from tqdm import tqdm
from math import factorial
from scipy.special import comb
import matplotlib.pyplot as plt
from itertools import combinations
from pyds import MassFunction
from shaded import detect_deceptive

n_x = 91
x_list = np.linspace(0, 0.9, n_x)

shapley_value_array = np.zeros((n_x, 3))
before_probabilities_array = np.zeros((n_x, 3))
after_probabilities_array = np.zeros((n_x, 3))


for x_index in tqdm(range(n_x)):
    x = round(x_list[x_index], 2)
    
    m1 = MassFunction({'a':0.6, 'b':0.2, 'c':0.1, 'abc':0.1})
    m2 = MassFunction({'a':0.5, 'b':0.3, 'c':0.1, 'abc':0.1})
    m3 = MassFunction({'a':x, 'b':0.9-x, 'c':0.1})
    
    mass_list = [m1, m2, m3] 
    n_mass = len(mass_list)
    n_states = 3
    
    shapley_values, strong_deceptive, weak_deceptive, p, t = detect_deceptive(mass_list, n_states)
    
    shapley_value_array[x_index,:] = shapley_values
    
    
    # evidence weighting
    m = mass_list[0]
    for i in range(1, len(mass_list)):
        m = m.combine_conjunctive(mass_list[i])
    before_p = m.pignistic()
    before_probabilities_array[x_index,0] = before_p['a']
    before_probabilities_array[x_index,1] = before_p['b']
    before_probabilities_array[x_index,2] = before_p['c']
    
    alphas = shapley_values.copy()
    non_deceptive_index = np.where(shapley_values>0)[0]
    non_deceptive_index = [0,1]
    non_deceptive_mass = []
    for i in range(len(non_deceptive_index)):
        non_deceptive_mass.append(mass_list[non_deceptive_index[i]+1].copy())
    
    m = non_deceptive_mass[0]
    for i in range(1, len(non_deceptive_index)):
        m = m.combine_conjunctive(non_deceptive_mass[i])
    after_p = m.pignistic()
    after_probabilities_array[x_index,0] = after_p['a']
    after_probabilities_array[x_index,1] = after_p['b']
    after_probabilities_array[x_index,2] = after_p['c']
    

x1_index = np.where(before_probabilities_array[:,0]<before_probabilities_array[:,1])[0][-1] + 1
x1 = x_list[x1_index] 
y1 = before_probabilities_array[x1_index,0]
# y1 = shapley_value_array[x1_index, -1]


# x2 = x_list[np.where(shapley_value_array[:,2]<0)[0][-1]+1]
# y2 = 0

x2_index = np.where(before_probabilities_array[:,0]<after_probabilities_array[:,0])[0][-1] +1
x2 = x_list[x2_index]
y2 = after_probabilities_array[x2_index,0]


# plt.rcParams["mathtext.fontset"] = "cm"

# figure 1
fig = plt.figure(figsize=(6,4))

plt.plot(x_list, before_probabilities_array[:,0], linewidth=2, ls='-.', color='tab:red', label='$p_a$')
plt.plot(x_list, before_probabilities_array[:,1], linewidth=2, ls=':', color='tab:blue', label='$p_b$')
plt.plot(x_list, before_probabilities_array[:,2], linewidth=2, ls='--', color='tab:green', label='$p_c$')

ax = plt.gca()
bottom, top = plt.ylim()
left, right =  plt.xlim()
ax.spines['bottom'].set_position(('data', bottom))
ax.spines['left'].set_position(('data', left))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot([x1, x1], [bottom, y1], linewidth=2, alpha=1, linestyle='-', color='gray', label='$x_1={}$'.format(x1))
plt.scatter(x1, y1, color='gray', zorder=2)

plt.xlabel('$x$', fontsize=10)
plt.ylabel('Pignistic Probabilities', fontsize=10)
plt.xticks(x_list[::10], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)

plt.savefig('figures/change_of_probabilities.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/change_of_probabilities.pdf', bbox_inches='tight', dpi=600)
plt.show()
plt.close()



# figure 2
fig = plt.figure(figsize=(6,4))

plt.plot(x_list, shapley_value_array[:,0], linewidth=2, ls=':', color='tab:blue', label='$φ_1$')
plt.plot(x_list, shapley_value_array[:,1], linewidth=2, ls='--', color='tab:green', label='$φ_2$')
plt.plot(x_list, shapley_value_array[:,2], linewidth=2, ls='-.', color='tab:red', label='$φ_3$')

ax = plt.gca()
bottom, top = plt.ylim()
left, right =  plt.xlim()
ax.spines['bottom'].set_position(('data', bottom))
ax.spines['left'].set_position(('data', left))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.xlabel('$x$', fontsize=10)
plt.ylabel('Shapley Values', fontsize=10)
plt.xticks(x_list[::10], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.4), fontsize=10) #, bbox_to_anchor=(0.2, 0.1, 0.8, 0.5)

plt.savefig('figures/change_of_shapley_values.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/change_of_shapley_values.pdf', bbox_inches='tight', dpi=600)
plt.show()
plt.close()


fig = plt.figure(figsize=(6,4))

plt.plot(x_list, before_probabilities_array[:,0], linewidth=2, ls='-.', color='tab:red', label='$p_a$ with $m_3$')
plt.plot(x_list, after_probabilities_array[:,0], linewidth=2, ls=':', color='tab:blue', label='$p_a$ without $m_3$')

ax = plt.gca()
bottom, top = plt.ylim()
left, right =  plt.xlim()
ax.spines['bottom'].set_position(('data', bottom))
ax.spines['left'].set_position(('data', left))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot([x2, x2], [bottom, y2], linewidth=2, alpha=1, linestyle='-', color='gray', label='$x_2={}$'.format(x2))
# plt.plot([left, x2], [y2, y2], linewidth=2, alpha=1, linestyle=':', color='gray')
plt.scatter(x2, y2, color='gray', zorder=2)

plt.xlabel('$x$', fontsize=10)
plt.ylabel('Pignistic Probabilities', fontsize=10)
plt.xticks(x_list[::10], fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.4), fontsize=10)

plt.savefig('figures/real_threshold.png', bbox_inches='tight', dpi=600)
plt.savefig('figures/real_threshold.pdf', bbox_inches='tight', dpi=600)
plt.show()
plt.close()