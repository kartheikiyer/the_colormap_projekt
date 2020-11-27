import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.colors import ListedColormap

from sklearn.manifold import TSNE
from sklearn import cluster, mixture

import colorsys
#import hilbert
from scipy.spatial import distance

import requests
import shutil

def get_image(url, imgtype='url', img_fname = 'cmap_image'):
    """
    Import image using matplotlib's mpimg module. 
    Specify either a url to the image, or a path on disk.
    """

    if imgtype == 'url':
        fname = url.split('/')[-1]
        try: 
            image = mpimg.imread('data/'+fname)
        except:
            print('image not found, downloading from URL')
            r = requests.get(url, stream=True, headers={'User-agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                with open("data/"+fname, 'wb') as f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, f)
            image = mpimg.imread('data/'+fname)
    elif imgtype == 'path':
        fname = url
        image = mpimg.imread(url)
    else:
        print('unknown imgtype: options are url or path.')
        
    if (fname[-3:] == 'jpg') or (fname[-4:] == 'jpeg'):
        image = image/255
        
    return image

def colors_lumsort(r,g,b):
    """
    Sort colors by luminosity
    """
    return np.sqrt( .241 * r + .691 * g + .068 * b )

def colors_e_sort(r,g,b):
    """
    Sort colors by E^2 value
    """
    lum = np.sqrt( .241 * r + .691 * g + .068 * b )  
    return np.sqrt(lum**2 + r**2 + b**2)

def colors_stepsort(r,g,b,repetitions=1):
    """
    Sort colors in hue steps for more perceptually uniform colormaps
    """
    lum = np.sqrt( .241 * r + .691 * g + .068 * b )
    h, s, v = colorsys.rgb_to_hsv(r,g,b)
    h2 = int(h * repetitions)
    lum2 = int(lum * repetitions)
    v2 = int(v * repetitions)
    if h2 % 2 == 1:
        v2 = repetitions - v2
        lum = repetitions - lum
    return (h2, lum, v2)

def colors_distsort(colours):
    """
    Sort colors by traveling salesman with euclidean distance
    """
    colours_length = len(colours)
    # Distance matrix
    A = np.zeros([colours_length,colours_length])
    for x in range(0, colours_length-1):
        for y in range(0, colours_length-1):
            A[x,y] = distance.euclidean(colours[x],colours[y])

    # Nearest neighbour algorithm
    path, _ = NN(A, 0)

    # Final array
    colours_nn = []
    for i in path:
        colours_nn.append( colours[i] )
    return colours_nn

def sort_colors(colorvals, sort_type = 'lum'):
    """
    Main function for sorting colors, specify type. Input and returns array of color values.
    
    Default: 'lum'
    
    Options are: 
    
    - rval: sort by just the R channel values
    - hsv: sort in hue, saturation, value space
    - hls: sort in hue, luminosity, saturation space
    - lum: sort by luminosity - great if you're interested in greyscaling plots
    - E: sort by E^2 value - weighted combination of lum, R, B
    - step: sort in hue steps for more perceptually uniform colormaps
    - hilbert: sort by hilbert stepping through space, currently not implemented
    - none: return as-is
    """
    colorlist = []
    for i in range(colorvals.shape[0]):
        colorlist.append((colorvals[i,0:].copy()))
    if sort_type == 'rval':
        color_order = np.argsort(colorvals,0)[0:,0]
        return colorvals[color_order,0:]
    elif sort_type == 'hsv':
        colorlist.sort(key=lambda rgb: colorsys.rgb_to_hsv(*rgb), reverse=True)
    elif sort_type == 'hls':
        colorlist.sort(key=lambda rgb: colorsys.rgb_to_hls(*rgb), reverse=True)
    elif sort_type == 'lum':
        colorlist.sort(key=lambda rgb: colors_lumsort(*rgb) )
    elif sort_type == 'E':
        colorlist.sort(key=lambda rgb: colors_e_sort(*rgb) )
    elif sort_type == 'step':
        colorlist.sort(key=lambda rgb: colors_stepsort(*rgb) )
    elif sort_type == 'hilbert':
        colorlist.sort(key=lambda rgb:hilbert.Hilbert_to_int([int(r*255),int(g*255),int(b*255)]) )
    elif sort_type == 'none':
        return colorvals
    else:
        print('Unknown sorting param [',sort_type,']: use rval, hsv, hls, lum, or step.')
    return np.array(colorlist)

def prune_colorset(colors, n_prune = 4, criterion = 'similar',metric = 'hls'):
    """
    remove colors that are too similar or too different from a colorset
    metric options are: 'hls','h', 'l'
    """
    

    colorlist_hls = []
    for i in range(colors.shape[0]):
        colorlist_hls.append((colorsys.rgb_to_hls(*colors[i,0:])))

    colorlist_hls = np.array(colorlist_hls)
    if metric == 'hls':
        color_dists = np.sum((colorlist_hls[1:,0:] - colorlist_hls[0:-1,0:])**2,1)
    elif metric == 'h':
        color_dists = ((colorlist_hls[1:,0] - colorlist_hls[0:-1,0])**2)
    elif metric == 'l':
        color_dists = ((colorlist_hls[1:,1] - colorlist_hls[0:-1,1])**2)
    else:
        print('unknown metric. use h, l, or hls')
        
    if criterion == 'similar':
        drop_args = np.argsort(color_dists)[0:n_prune]
    elif criterion == 'different':
        drop_args = np.argsort(color_dists)[-n_prune:]
    else:
        print('unknown criterion')
    #print('dropping colors: ',drop_args)
    colors_pruned = np.delete(colors,drop_args, axis=0)
    
    return colors_pruned

def plt_image(image):
    
    plt.figure(figsize=(image.shape[0]/100, image.shape[1]/100))
    plt.imshow(image)
    plt.show()
    return

def get_colors_from_image(image, ncolors = 6, npts = 1000, rseed = 42, use_tsne = True, tsne_perp = 40, tsne_eggr = 12, dbscan_eps = 0.3, verbose = True, clustering = 'kmeans', sort_type='lum', n_prune = 0, prune_metric = 'hls'):
    """
    Main function for taking in an input image and returning a set of colors+colormap for use in plots. 
    This works by taking an image and, 
    1. running t-SNE on it to redistribute colors more uniformly in features space (optional), followed by
    2. running clustering on them to identify different regions and computing the median color in each, and then
    3. ordering the colors to create a colormap.
    
    To see if methods are working as intended (i.e. output colors are representative of input image), 
    run in verbose mode and play around with the ncolors, tsne_perp and clustering arguments. 
    
    Arguments:
    - image: [M x N x 3] array of RGB values (scaled to [0,1]), usually output of get_image() function.
    - ncolors: [default 6] number of colors to output. Not used if clustering == 'DBSCAN'.
    - npts: [default 1000] number of points to sample from image. set higher if image has a high dynamic range.
    - rseed: [default 42] random seed for tSNE + clustering
    - use_tsne: [default True] flag that decides whether to run t-SNE prior to clustering
    - tsne_perp: [default 40] t-SNE perplexity
    - tsne_eggr: [default 12] t-SNE early exaggeration
    - dbscan_eps: [default 0.3] eps parameter for DBSCAN. not used unless clustering == 'DBSCAN'
    - verbose: [default False] set to True to see verbose output of color identification process
    - clustering: [default kmeans] type of clustering to identify median colors. Options are 'kmeans','DBSCAN','GMM', 'optics', and 'spectral'.
    - n_prune: [default 0] - prune colorset to remove similar colors in hue or hls space.
    - prune_metric: [default 'hls'] - metric to prune colors in
    """
    
    np.random.seed(rseed)
    
    if verbose == True:
        plt.figure(figsize=(image.shape[0]/100, image.shape[1]/100))
        plt.imshow(image)
        plt.show()

    cvs = image.reshape(image.shape[0]*image.shape[1],3)[np.random.choice(image.shape[0]*image.shape[1], size=npts),0:]

    if use_tsne == True:
        if verbose == True:
            tsne = TSNE(n_components=2, verbose=1, perplexity=tsne_perp, n_iter=300, early_exaggeration=tsne_eggr)
        else:
            tsne = TSNE(n_components=2, perplexity=tsne_perp, n_iter=300, early_exaggeration=tsne_eggr)
        tsne_results = tsne.fit_transform(cvs)
    else:
        tsne_results = cvs

    if clustering == 'kmeans':
        clust = cluster.KMeans(n_clusters=ncolors,init='k-means++')
        clust.fit(tsne_results)
        
    elif clustering == 'DBSCAN':
        clust = cluster.DBSCAN(eps=dbscan_eps, min_samples=int(npts/ncolors/10))
        clust.fit(tsne_results)
        ncolors = len(set(clust.labels_)) - (1 if -1 in clust.labels_ else 0)
        print('# colors from DBSCAN = ',str(ncolors))
        
    elif clustering == 'optics':
        clust = cluster.OPTICS(min_samples=int(npts/ncolors/10), xi = dbscan_eps, min_cluster_size = int(npts/ncolors/10))
        clust.fit(tsne_results)
        
    elif clustering == 'spectral':
        clust = cluster.SpectralClustering(
        n_clusters=ncolors, eigen_solver='arpack',
        affinity="nearest_neighbors")
        clust.fit(tsne_results)
    
    elif clustering == 'GMM':
        clust = mixture.GaussianMixture(n_components=ncolors, covariance_type='full')
        clust.fit(tsne_results)
        clust.labels_ = clust.predict(tsne_results)

    # plt.hist(kmeans.labels_, np.arange(ncolors+1)-0.5)
    # plt.show()
    if verbose == True:
        plt.figure(figsize=(14,6))
        plt.subplot(1,2,1)
        plt.scatter(tsne_results[0:,0], tsne_results[0:,1],s=100, c=cvs)
        if use_tsne == True:
            plt.axis([-20,20,-20,20]);
            plt.title('t-SNE of color space')
        else:
            plt.title('distribution of color space')
        plt.xticks([]);plt.yticks([])
        plt.subplot(1,2,2)
        plt.scatter(tsne_results[0:,0], tsne_results[0:,1],s=100, c=clust.labels_)
        if use_tsne == True:
            plt.axis([-20,20,-20,20]);
        plt.xticks([]);plt.yticks([])
        plt.title('color clusters')
        plt.show()

    colorvals = np.zeros((ncolors,3))
    for i in range(ncolors):
        colorvals[i,0:3] = np.median(cvs[clust.labels_==i,0:],0)
        
    cvals = sort_colors(colorvals, sort_type=sort_type)
    cvals = prune_colorset(cvals.copy(),n_prune=n_prune,metric=prune_metric)
    cmap = ListedColormap(sort_colors(cvals, sort_type=sort_type))
    
    if verbose == True:
        plt_cmap(cvals,sort = False)

    return cvals, cmap

def plt_cmap(colorvals,marker = 's',size = 28000, sort = True, sort_type = 'lum'):
    """
    Plot colormap inferred from input image
    """
    
    ncolors = colorvals.shape[0]
    if sort == True:
        sorted_colorvals = sort_colors(colorvals, sort_type=sort_type)
    else:
        sorted_colorvals = colorvals
    plt.figure(figsize=(ncolors*3,3))
    plt.scatter(np.arange(ncolors),np.zeros((ncolors,)), c=sorted_colorvals,s=size,marker=marker)
    plt.axis([-0.5,ncolors-0.5,-0.5,0.5])
    plt.xticks([]);plt.yticks([])
    plt.show()
    
    return

def plot_examples(colors, cmap, xlen=50, ylen = 30):
    """
    Plot examples showing the colormap at work
    """
    
    np.random.seed(19680801)
    Z = np.random.rand(ylen, xlen)
    x = np.linspace(-0.5, 10, xlen+1)  # len = 11
    y = np.linspace(4.5, 11, ylen+1)  # len = 7
    xt, yt = np.meshgrid(x[0:-1],y[0:-1])
    Z = Z+2*np.exp(-((np.sqrt(xt*xt+yt*yt) - 10)**2)/1.0**2)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.pcolormesh(x, y, Z, cmap=cmap)
    plt.colorbar()
    plt.subplot(2,1,2)
    nlines = colors.shape[0]
    tax = np.arange(0,10,0.01)
    for i in range(nlines):
        plt.plot(tax, np.sin(tax*(i+3)/5),lw=3,c = colors[i,0:])
    plt.tight_layout()
    plt.show()