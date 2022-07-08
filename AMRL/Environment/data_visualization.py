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


def plot_graph(reward: np.array,
               precision: np.array,
               alpha: np.array,
               episode_len: np.array,
               avg_rewards: np.array,
               avg_alphas: np.array,
               avg_precisions: np.array,
               avg_episode_lengths: np.array) -> None:
    """
    Plot training progress (reward, precision, alpha, episode length and their mean)

    Parameters
    ---------
    reward, precision, alpha, episode_len, avg_rewards, avg_alphas, avg_precisions, avg_episode_lengths: array_like

    Returns
    -------
    None : None
    """
    clear_output(wait=True)
    _, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12,8))
    ax_plot_graph(reward, avg_rewards, ax, 'Episode reward')
    #ax.set_ylim([-8, np.max(avg_rewards)+1])
    ax_plot_graph(precision, avg_precisions, ax1, 'Precision (nm)')
    ax1.set_ylim([0, 1])
    ax1.hlines(0.166, 0, len(precision))
    ax_plot_graph(alpha, avg_alphas, ax2, 'alpha')
    ax_plot_graph(episode_len, avg_episode_lengths, ax3, 'Episode lengths')
    plt.show()


def ax_plot_graph(data: np.array,
                  avg_data: np.array,
                  ax: matplotlib.axes.Axes,
                  y_label: str) -> None:
    """
    Plot data and its rolling mean

    Parameters
    ---------

    data, avg_data : array_like

    ax : axes

    y_label : str

    Returns
    -------

    None : None

    """
    df_dict = {'x': range(len(data)), 'data': data, 'average': avg_data}
    df = pd.DataFrame(df_dict)
    args = df['x'], df['data']
    ax.plot(*args, marker='', color='silver', linewidth=0.8, alpha=0.9)
    args = df['x'], df['average']
    ax.plot(*args, marker='', color='DodgerBlue', linewidth=1, alpha=0.9)
    ax.set_xlabel("episode", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)


def show_reset(img_info: dict,
               atom_start_position: np.array,
               destination_position: np.array,
               template_nm: np.array = None,
               template_wh: np.array = None) -> None:
    """
    Show STM image, atom and target position, and template position at the reset step in reinforcement learning

    Parameters
    ---------

    img_info : dict

    atom_start_position, destination_position : array_like
        atom and target position in STM coordinate (nm)

    template_nm, template_wh: array_like, optional
        template position and size in nm

    Returns
    -------
    None : None
    """
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()

    e1 = offset_nm[0]-0.5*len_nm[0]
    e2 = offset_nm[0]+0.5*len_nm[0]
    e3 = offset_nm[1]+len_nm[0]
    e4 = offset_nm[1]
    extent = (e1, e2, e3, e4)
    ax.imshow(img, extent=extent)
    if (template_nm is not None) and (template_wh is not None):
        args = template_nm,template_wh[0], template_wh[1]
        kwargs = {'linewidth':1, 'edgecolor':'r', 'facecolor':'none'}
        rect = matplotlib.patches.Rectangle(*args, **kwargs)
        ax.add_patch(rect)

    args = atom_start_position[0], atom_start_position[1]
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#33dbff', 'color':None, 'label':'start'}
    ax.scatter(*args,**kwargs)

    args = destination_position[0], destination_position[1]
    kwargs =  {'s':20, 'linewidths':3, 'edgecolors':'#75ff33', 'color': None, 'label':'goal'}
    ax.scatter(*args,**kwargs)
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()


def show_done(img_info: dict,
              atom_position: np.array,
              atom_start_position: np.array,
              destination_position: np.array,
              reward: float,
              template_nm: np.array = None,
              template_wh: np.array = None) -> None:
    """
    Show STM image, atom start and current position, target position,
    episode reward, and template position when RL episode terminates

    Parameters
    ---------
    img_info : dict

    atom_position,atom_start_position, destination_position: array_like
        atom start, atom current, and target position in STM coordinate (nm)

    reward: float
        episode reward

    template_nm, template_wh: array_like, optional
        template position and size in nm

    Returns
    -------

    None : None

    """
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()
    e1 = offset_nm[0]-0.5*len_nm[0]
    e2 = offset_nm[0]+0.5*len_nm[0]
    e3 = offset_nm[1]+len_nm[0]
    e4 = offset_nm[1]
    extent = (e1, e2, e3, e4)
    ax.imshow(img, extent = extent)
    if (template_nm is not None) and (template_wh is not None):
        args = template_nm,template_wh[0], template_wh[1]
        kwargs = {'linewidth':1, 'edgecolor':'r', 'facecolor':'none'}
        rect = matplotlib.patches.Rectangle(*args, **kwargs)
        ax.add_patch(rect)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#33dbff', 'color':None, 'label':'start'}
    ax.scatter(atom_start_position[0], atom_start_position[1], **kwargs)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#75ff33', 'color': None, 'label':'goal'}
    ax.scatter(destination_position[0], destination_position[1], **kwargs)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#ff5733', 'color':None, 'label':'atom'}
    ax.scatter(atom_position[0], atom_position[1], **kwargs)

    ax.text(offset_nm[0], offset_nm[1],'reward: {}'.format(reward), ha='center')
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()


def show_step(img_info: dict,
              start_nm: np.array,
              end_nm: np.array,
              atom_position: np.array,
              atom_start_position: np.array,
              destination_position: np.array,
              mvolt: float,
              pcurrent: float,
              template_nm: np.array = None,
              template_wh: np.array = None) -> None:
    """Show STM image, atom start and current position, target position,
    bias, current setpoint, and template position when environment takes RL step

    Parameters
    ---------

    img_info : dict

    atom_position,atom_start_position, destination_position : array_like
        atom start, atom current, and target position in STM coordinate (nm)

    mvolt, pcurrent : float
        bias in mV and current in pA

    template_nm, template_wh : array_like, optional
        template position and size in nm

    Returns
    -------

    None : None

    """
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    _, ax = plt.subplots()
    e1 = offset_nm[0]-0.5*len_nm[0]
    e2 = offset_nm[0]+0.5*len_nm[0]
    e3 = offset_nm[1]+len_nm[0]
    e4 = offset_nm[1]
    extent = (e1, e2, e3, e4)
    ax.imshow(img, extent=extent)
    if (template_nm is not None) and (template_wh is not None):
        args = template_nm,template_wh[0], template_wh[1]
        kwargs = {'linewidth':1, 'edgecolor':'r', 'facecolor':'none'}
        rect = matplotlib.patches.Rectangle(*args, **kwargs)
        ax.add_patch(rect)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#33dbff', 'color':None, 'label':'start'}
    ax.scatter(atom_start_position[0], atom_start_position[1], **kwargs)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#ff5733', 'color':None, 'label':'atom'}
    ax.scatter(atom_position[0], atom_position[1], **kwargs)
    kwargs = {'s':20, 'linewidths':3, 'edgecolors':'#75ff33', 'color':None, 'label':'goal'}

    ax.scatter(destination_position[0], destination_position[1], **kwargs)
    x = start_nm[0]
    y = start_nm[1]
    dx = (end_nm - start_nm)[0]
    dy = (end_nm - start_nm)[1]
    ax.arrow(x, y, dx, dy, width=0.1, length_includes_head=True)
    txt_x = offset_nm[0] + 0.5*len_nm[0]
    txt_y = offset_nm[1] + len_nm[0]
    text = 'bias(mV):{:.2f}, current(nA):{:.2f}'.format(mvolt,pcurrent/1000)
    ax.text(txt_x, txt_y, text)
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()


def plot_large_frame(img_info: dict,
                     atom_chosen: np.array,
                     design_chosen: np.array,
                     anchor_chosen: np.array,
                     next_destination: np.array,
                     path: np.array) -> None:
    """
    Used for building multiple atom structures
    Show STM image, atoms, designs, and anchor, next target, and path between atom and design

    Parameters
    ---------
    img_info : dict

    atom_chosen, design_chosen, anchor_chosen, next_destination : array_like
        atom, design, anchor, and next target positions in STM coordinate (nm)

    path : array_like
        path between atom and design

    Returns
    -------
    None : None
    """
    _, ax = plt.subplots()
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    e1 = offset_nm[0]-0.5*len_nm[0]
    e2 = offset_nm[0]+0.5*len_nm[0]
    e3 = offset_nm[1]+len_nm[0]
    e4 = offset_nm[1]
    extent = (e1, e2, e3, e4)
    ax.imshow(img, extent=extent)
    ax.scatter(atom_chosen[0], atom_chosen[1], color='#7027A0', label='atom')
    ax.scatter(design_chosen[0], design_chosen[1], color='#1DB9C3', label='design')
    ax.scatter(anchor_chosen[0], anchor_chosen[1], color='#F56FAD', label='anchor')

    path = np.array(path)
    ax.plot(path[:,0], path[:,1])
    x, y = atom_chosen[0],atom_chosen[1]
    dx = (next_destination - atom_chosen)[0]
    dy = (next_destination - atom_chosen)[1]
    ax.arrow(x, y, dx, dy, width=0.1,length_includes_head=True, color='#FFC069')
    ax.legend(frameon = False, labelcolor = '#FAEBE0')
    plt.show()


def plot_atoms_and_design(img_info: dict,
                          all_atoms: np.array,
                          design: np.array,
                          anchor: np.array,
                          show_legend: bool=True) -> None:
    """Plot the atoms and design pattern

    Parameters
    ----------

    img_info : dict

    all_atoms : array_like

    design : array_like

    anchor : array_like

    show_legend : bool

    Returns
    -------

    None

    """
    _, ax = plt.subplots()
    img = img_info['img_forward']
    offset_nm = img_info['offset_nm']
    len_nm = img_info['len_nm']
    e1 = offset_nm[0]-0.5*len_nm[0]
    e2 = offset_nm[0]+0.5*len_nm[0]
    e3 = offset_nm[1]+len_nm[0]
    e4 = offset_nm[1]
    extent = (e1, e2, e3, e4)
    ax.imshow(img, extent=extent, cmap='Greys')
    if (all_atoms is not None) and (all_atoms.size !=0):
        ax.scatter(all_atoms[:,0], all_atoms[:,1], color='#7027A0' ,label='atom')
    if design is not None:
        ax.scatter(design[:,0],design[:,1], color='#1DB9C3', label='design')
    if anchor is not None:
        ax.scatter(anchor[0], anchor[1], color = '#F56FAD', label='anchor')
    if show_legend:
        ax.legend(frameon = False, labelcolor = '#FAEBE0')
    plt.show()
