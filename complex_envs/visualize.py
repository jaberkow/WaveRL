import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import yaml
import argparse
import os
sns.set_style('white')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='rollout_file',
        help='Path to the npz outputfile of a rollout',default='', type=str)
    parser.add_argument('-f',dest='output_prefix',
        help='Output visualization filenames start with this',default='output',type=str)
    args = parser.parse_args()

    #load in the numpy data
    data = np.load(args.rollout_file)

    #unpack the data
    u_array = data['u_array']
    impulse_array = data['impulse_array']
    energy_array = data['energy_array']
    x_mesh = data['x_mesh']
    code_array = data['code_array']

    #animation of demo
    x_min = x_mesh[0]
    x_max = x_mesh[-1]

    y_min = min([np.min(u_array),np.min(impulse_array)])
    y_max = max([np.max(u_array),np.max(impulse_array)])
    frame_num = np.shape(u_array)[1]

    fig = plt.figure()
    ax1 = plt.axes(xlim=(x_min, x_max), ylim=(y_min,y_max))
    line, = ax1.plot([], [], lw=2)
    plt.xlabel('Bridge Position')
    plt.ylabel('Height')

    plotlays, plotcols = [3], ["red","black","green"]
    plotlegends = ['bridge','','impulse']
    lines = []
    for index in range(3):
        lobj = ax1.plot([],[],lw=2,color=plotcols[index],label=plotlegends[index])[0]
        lines.append(lobj)
    ax1.legend(loc='upper left')

    def init():
        for line in lines:
            line.set_data([],[])
        return lines




    def animate(i):

        xlist = [x_mesh, x_mesh,x_mesh]
        ylist = [u_array[:,i], np.zeros_like(u_array[:,i]),impulse_array[:,i]]
        if code_array[i]==0:
            plt.title('Step {} (warmup)'.format(int(i)))
        elif code_array[i]==1:
            plt.title('Step {} (equilibriate)'.format(int(i)))
        elif code_array[i]==2:
            plt.title('Step {} (dampen)'.format(int(i)))
        #for index in range(0,1):
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately. 

        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frame_num,interval=100, blit=True)

    Writer = animation.writers['pillow']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim_outname = args.output_prefix + '.gif'
    anim.save(anim_outname, writer=writer)
    plt.clf()
    #plot a graph of the energy over the episode

    warmup_energy = []
    warmup_step = []

    equi_energy = []
    equi_step= []

    dampen_energy = []
    dampen_step = []

    for i in range(len(energy_array)):
        if code_array[i]==0:
            warmup_energy.append(energy_array[i])
            warmup_step.append(i)
        if code_array[i]==1:
            equi_energy.append(energy_array[i])
            equi_step.append(i)
        if code_array[i]==2:
            dampen_energy.append(energy_array[i])
            dampen_step.append(i)

    plt.plot(warmup_step,warmup_energy,'r',label='warmup')
    plt.plot(equi_step,equi_energy,'g',label='equilibriate')
    plt.plot(dampen_step,dampen_energy,'b',label='dampen')


    plt.legend(loc='upper left')

    plt.xlabel('step',fontsize=15)
    plt.ylabel('energy',fontsize=15)
    trend_outname = args.output_prefix + '.eps'
    plt.savefig(trend_outname)
