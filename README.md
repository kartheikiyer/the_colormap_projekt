# `the_colormap_projekt`
Creating colormaps from pictures with tiny bits of ML

# The Colormap Projekt aims to create clean, visually appealing colormaps from input data

Currently the formats are limited to images, but there are plans to improve this to include videos and other media in the future. Open an issue if you're interested in collaborating on this. 

The way this works is:

1. **The image is broken down into RGB color space**, with sparse sampling for computational efficiency (controlled via the `npts` argument).
2. **The colors are then (optionally) embedded in 2 dimensions using t-SNE**. (See Laurens van der Maaten's [page](https://lvdmaaten.github.io/tsne/) for more information). This is done mainly to get the colors prepped for clustering, (although t-SNE prior to clustering is not always recommended). I find that for this particular application, the resizing of clusters on running tSNE makes things more uniform in color space, and therefore better suited to build palettes. tSNE perplexity and early exaggeration are controlled via the `tsne_perp` and `tsne_eggr` parameters. This step can be skipped by turning off the `use_tsne` boolean flag. 
3. **The tSNE embedding is then split into `ncolors` discrete clusters**, with the clustering implemented through a variety of methods (`clustering` = kmeans, GMM, DBSCAN, spectral, optics from sklearn) and the median color in each cluster is calculated. 
4. **The colors are then sorted** according to different color sorting mechanisms (`sort_type` = rval, hsv, hls, lum, E, step. See `sort_colors()` for more). This can make the palette more uniform in color space, either perceptually, or individual in hue or luminosity.
5. Finally, **excess colors are pruned** using the `n_prune` and `prune_metric` arguments. This essentially can be used to filter out dominant background colors etc. that might otherwise swamp the palette. A combination of high `ncolors` and `n_prune` can be used to select for bright, sharp colors that are only in a small part of the image space.

## To use the method out of the box without tuning any of the parameters takes just three steps!

1. get the image in python (with a url or image path) using `cmp.get_image(url)`. Images are stored in the `data/` folder.
2. generate the colormap using `colors, cmap = cmp.get_colors_from_image(image, ncolors, n_prune)`. This returns both colors that can be used directly for lines etc. or a cmap object that can be used to color contours, 2d/3d plots and more.
3. visualise the image and colormap using `cmp.plt_image(image);cmp.plt_cmap(colors);` That's it!

There're a few demo images below, but feel free to try this out with your own images at the end of this tutorial (if you're running it on google colab). Play around with the parameters till you find something that works best for you! If you create any cool palettes using this, please attribute this package, and let me know!
