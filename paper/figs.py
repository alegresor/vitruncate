from vitruncate import GT
from numpy import *
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
pyplot.rc('font', size=16)  #set defaults so that the plots are readable
pyplot.rc('axes', titlesize=16)
pyplot.rc('axes', labelsize=16)
pyplot.rc('xtick', labelsize=16)
pyplot.rc('ytick', labelsize=16)
pyplot.rc('legend', fontsize=16)
pyplot.rc('figure', titlesize=16)

def plot_pdf():
    gt_1d_pdf = lambda x, std, beta: 1/(std*sqrt(2*pi))*exp(-x**2/(2*std**2))*(x<=1)*(x>=-1) + (x>1)*1/(x**beta) + (x<-1)*1/((-x)**beta)
    fig,ax = pyplot.subplots()
    x = linspace(-3,3,num=5000,endpoint=True)
    y_1 = gt_1d_pdf(x,std=1,beta=5)
    y_2 = gt_1d_pdf(x,std=.5,beta=4)
    ax.plot(x,y_1,color='c',label=r'$\tilde{\mathcal{N}}(0,1,\beta=5)$')
    ax.plot(x,y_2,color='g',label=r'$\tilde{\mathcal{N}}(0,1/4,\beta=4)$')
    ax.set_xlim([-3,3])
    ax.set_xticks([-3,0,3])
    ax.set_ylim([0,1])
    ax.set_yticks([0,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, fancybox=True, shadow=True)
    pyplot.savefig('paper/figs/pdf.png',dpi=250)

def plot_cub_ext():
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = meshgrid([0,1], [0,1])
    ax.plot_surface(X,Y,ones((2,2)), color='b')
    ax.plot_surface(X,Y,zeros((2,2)), color='b')
    ax.plot_surface(X,zeros((2,2)),Y, color='b')
    ax.plot_surface(X,ones((2,2)),Y, color='b')
    ax.plot_surface(ones((2,2)),X,Y, color='b')
    ax.plot_surface(zeros((2,2)),X,Y, color='b')
    ext = 'faces'
    if 'lines' in ext:
        ax.plot([-1,2],[0,0],[0,0],color='r')
        ax.plot([-1,2],[1,1],[0,0],color='r')
        ax.plot([-1,2],[0,0],[1,1],color='r')
        ax.plot([-1,2],[1,1],[1,1],color='r')
        ax.plot([0,0],[-1,2],[0,0],color='r')
        ax.plot([1,1],[-1,2],[0,0],color='r')
        ax.plot([0,0],[-1,2],[1,1],color='r')
        ax.plot([1,1],[-1,2],[1,1],color='r')
        ax.plot([0,0],[0,0],[-1,2],color='r')
        ax.plot([1,1],[0,0],[-1,2],color='r')
        ax.plot([0,0],[1,1],[-1,2],color='r')
        ax.plot([1,1],[1,1],[-1,2],color='r')
    if 'faces' in ext:
        Xe, Ye = meshgrid([-1,2], [-2,2])
        ax.plot_surface(Xe,Ye,ones((2,2)), color='b',alpha=.5)
        #ax.plot_surface(Xe,Ye,zeros((2,2)), color='b',alpha=.5)
        ax.plot_surface(Xe,zeros((2,2)),Ye, color='g',alpha=.5)
        #ax.plot_surface(Xe,ones((2,2)),Ye, color='g',alpha=.5)
        ax.plot_surface(ones((2,2)),Xe,Ye, color='r',alpha=.5)
        #ax.plot_surface(zeros((2,2)),Xe,Ye, color='r',alpha=.5)
    if 'points' in ext:
        points = array([[0, 0, 0],[1, 0, 0 ],[1, 1, 0],[0, 1, 0],[0, 0, 1],[1, 0, 1 ],[1, 1, 1],[0, 1, 1]])
        ax.scatter3D(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=0)
    pyplot.savefig('paper/figs/cube_ext.png',dpi=250)

def plot_square_ext():
    fig,ax = pyplot.subplots(figsize=(5,5))
    ax.add_patch(patches.Rectangle((-.5,-.5),1,1,linewidth=0,edgecolor='b',facecolor='c'))
    ax.plot([-2,2],[-.5,-.5],color='g')
    ax.plot([-2,2],[.5,.5],color='g')
    ax.plot([-.5,-.5],[-2,2],color='g')
    ax.plot([.5,.5],[-2,2],color='g')
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_aspect(1)
    for s in ['left','right','top','bottom']:
        ax.spines[s].set_visible(False)
    pyplot.axis('off')
    adj = -.05
    pyplot.text(-1+adj, -1+adj, '2')
    pyplot.text(-1+adj, 1+adj, '2')
    pyplot.text(1+adj, 1+adj, '2')
    pyplot.text(1+adj, -1+adj, '2')
    pyplot.text(0+adj, -1+adj, '1')
    pyplot.text(0+adj, 1+adj, '1')
    pyplot.text(-1+adj, 0+adj, '1')
    pyplot.text(1+adj, 0+adj, '1')
    pyplot.text(0+adj, 0+adj, '0')
    pyplot.tight_layout()
    pyplot.savefig('paper/figs/square_ext.png',dpi=250)

def heatmap():
    # params
    L = [-2,-4]
    U = [4,5]
    n_mesh = 100
    pdelta = 1
    xlim = [L[0]-pdelta,U[0]+pdelta]
    ylim = [L[1]-pdelta,U[1]+pdelta]
    # generate points
    gt = GT(
        n = 2**8, 
        d = 2,
        mu = [1,2], 
        Sigma = [[5,4],[4,9]], #[[5,0],[0,9]],
        L = L, 
        U = U, 
        init_type = 'Sobol',
        seed = None,
        n_block = None,
        alpha=.1)
    x = gt.update(steps=1000, epsilon=5e-3, eta=.9)
    # evaluate meshgrid for pdf contour
    mesh = zeros(((n_mesh)**2,3),dtype=float)
    x_grid_tics = linspace(xlim[0],xlim[1],n_mesh)
    y_grid_tics = linspace(ylim[0],ylim[1],n_mesh)
    x_mesh,y_mesh = meshgrid(x_grid_tics,y_grid_tics)
    mesh[:,0] = x_mesh.flatten()
    mesh[:,1] = y_mesh.flatten()
    mesh[:,2] = log2(gt._pdf(mesh[:,:2]))
    z_mesh = mesh[:,2].reshape((n_mesh,n_mesh))
    # plots
    fig,ax = pyplot.subplots(figsize=(5,5))
    #   colors 
    clevel = linspace(mesh[:,2].min(),mesh[:,2].max(),100)
    cmap = pyplot.get_cmap('GnBu') # https://matplotlib.org/tutorials/colors/colormaps.html
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(.95,.95,.95),(0,0,1)])
    #   contours
    ax.contourf(x_mesh,y_mesh,z_mesh,clevel,cmap=cmap,extend='both')
    #ax.contour(x_mesh,y_mesh,z_mesh,levels=[-50,-30,-10,-1])
    #   scatter plot 
    ax.scatter(x[:,0],x[:,1],s=5,color='w')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([xlim[0],L[0],U[0],xlim[1]])
    ax.set_yticks([ylim[0],L[1],U[1],ylim[1]])
    ax.set_title(r'Density Log Contour with $\alpha$=.1')
    # output
    pyplot.savefig('paper/figs/heatmap.png',dpi=250) 

if __name__ == '__main__':
    #plot_pdf()
    #plot_cub_ext()
    #plot_square_ext()
    heatmap()