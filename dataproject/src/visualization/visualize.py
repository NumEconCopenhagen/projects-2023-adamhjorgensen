import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_delta(data):
    # a. Set up sector list and names
    sectors = list(set(data.index.get_level_values('BRANCHE')))
    sector_names = {'ene':'Energy',
                    'off':'Public sector', 
                    'soe':'Transport by sea', 
                    'byg':'Construction', 
                    'tje':'Services', 
                    'udv':'Extraction', 
                    'fre':'Manufacturing', 
                    'lan':'Agriculture', 
                    'bol':'Households'}

    # b. make plot
    fig, axs = plt.subplots(3, 3)

    count = 0
    for sec in sectors:
        # i. Make ax indecies
        i = count % 3
        j = count // 3
        
        # ii. Plot delta for machine and buildings
        y = data.loc[sec]
        if count == 0: #Add label once only
            axs[i, j].plot(y, label=['Building capital', 'Machine capital'])
        else: 
            axs[i, j].plot(y)
            
        # iii. Set sector title 
        axs[i, j].set_title(sector_names[sec])
        
        # iv. increase counter
        count +=1

    # c. figure settings
    fig.tight_layout()
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.0),
            fancybox=True, shadow=True, ncol=5)