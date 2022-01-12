from matplotlib import pyplot as plt
from IPython.display import clear_output
import pandas as pd
import matplotlib
import numpy as np

def circle(x, y, r, p = 100):
    x_, y_ = [], []
    for i in range(p):
        x_.append(x+r*np.cos(2*i*np.pi/p))
        y_.append(y+r*np.sin(2*i*np.pi/p))
    return x_, y_

def plot_graph(reward, precision, alpha, episode_len):
    clear_output(wait=True) 
    _, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    ax_plot_graph(reward, ax, 'Episode reward')
    ax_plot_graph(precision, ax1, 'Precision (nm)')
    ax_plot_graph(alpha, ax2, 'alpha')
    ax_plot_graph(episode_len, ax3, 'Episode lengths')
    plt.show()

def ax_plot_graph(data, ax, y_label):
    avg_data = np.mean(data[-min(100, len(data)):])
    df = pd.DataFrame({'x': range(len(data)), 'data': data, 'average': avg_data})
    ax.plot(df['x'], df['data'], marker='', color='silver', linewidth=0.8, alpha=0.9)
    ax.plot(df['x'], df['average'], marker='', color='DodgerBlue', linewidth=1, alpha=0.9)
    ax.set_xlabel("episode", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    

def show_reset(img_info, atom_start_position, destination_position, template_nm = None, template_wh = None):
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])

    ax.imshow(img, extent = extent)
    if (template_nm is not None) and (template_wh is not None):
        rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()

def show_done(img_info, atom_position, atom_start_position, destination_position, reward, new_destination_absolute_nm = None, template_nm = None, template_wh = None):
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    if (template_nm is not None) and (template_wh is not None):
        rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    ax.scatter(atom_position[0], atom_position[1], s = 20, linewidths=3, edgecolors='#ff5733', color = None, label='atom')

    if new_destination_absolute_nm is not None:
        ax.scatter(new_destination_absolute_nm[0], new_destination_absolute_nm[1], s = 20, linewidths=3, edgecolors='gray', color = 'gray', label='new destination')

    ax.text(offset_nm[0], offset_nm[1],'reward: {}'.format(reward), ha='center')
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()

def show_step(img_info, start_nm, end_nm, atom_position, atom_start_position, destination_position, mvolt, pcurrent,
              template_nm = None, template_wh = None):
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    if (template_nm is not None) and (template_wh is not None):
        rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(atom_position[0], atom_position[1], s = 20, linewidths=3, edgecolors='#ff5733', color = None, label='atom')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    ax.arrow(start_nm[0], start_nm[1], (end_nm - start_nm)[0], (end_nm - start_nm)[1],width=0.1, length_includes_head = True)
    ax.text(offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], 'bias(mV):{:.2f}, current(nA):{:.2f}'.format(mvolt,pcurrent/1000))
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()
    
    
def plot_large_frame(img, offset_nm, len_nm, design, start, i):
    fig, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    for g in design:
            x, y = circle(g[0], g[1], r)
            ax.plot(x,y, color='#A45D5D')
    for g in start:
        x, y = circle(g[0], g[1], r)
        ax.plot(x,y, color='#4A403A')
        
    for j in range(start.shape[0]):
        plt.arrow(start[j,0],start[j,1], (design-start)[j,0], (design-start)[j,1], width=0.1,length_includes_head=True, color='#FFC069')
        fs = 14
        if j==i:
            fs = 20
        ax.annotate(j, (start[j,0], start[j,1]), fontsize=fs)
    plt.show()

