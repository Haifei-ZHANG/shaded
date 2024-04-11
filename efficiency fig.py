# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:45:04 2024

@author: zhanghai
"""

import matplotlib.pyplot as plt

# Data
x = [5, 10, 15, 20]
y1 = [0.245, 8.937, 78.498, 312.568]
y2 = [0.001, 0.064, 1.783, 78.287]

plt.rcParams.update({'font.size': 14})
# Create the plot
plt.figure(figsize=(10, 4))

# Plot y1 data
plt.plot(x, y1, label="Kang's method", marker='o', linestyle='-', linewidth=3, color='tab:blue')
# Annotate each y1 value
for i, txt in enumerate(y1):
    plt.annotate(f"{txt:.3f}", (x[i], y1[i]), textcoords="offset points", xytext=(-15*(i>1),5), ha='center')

# Plot y2 data
plt.plot(x, y2, label='Our method', marker='s', linestyle='--', linewidth=3, color='tab:red')
# Annotate each y2 value
for i, txt in enumerate(y2):
    plt.annotate(f"{txt:.3f}", (x[i], y2[i]), textcoords="offset points", xytext=(15*(i>2),-15), ha='center')

plt.xticks(x)
plt.ylim(bottom=-max(y1+y2)*0.1, top=max(y1+y2)*1.1)
plt.xlim(right=21)


# Add legend
plt.legend(fontsize=14)


plt.xlabel('number of mass functions',fontsize=14)
plt.ylabel('elapsed time (seconds)',fontsize=14)

# Display the plot
plt.grid(True)
plt.tight_layout()

# plt.savefig("figures/efficiency.esp", dpi=600)
plt.savefig("figures/efficiency.png", dpi=600)
plt.show()