import numpy as np


def plot_seg_and_fofs(seg, fofs, width=1000, rng=None, show=False, **kw):
    """
    plot the fof groups over the seg map

    Parameters
    ----------
    seg: array
        seg map
    fofs: array with fields
        An array with fof_id, x, y fields.  Note the output of get_fofs() does
        not include x, y so you must add those yourself
    width: int, optional
        Image width in pixels
    rng: np.random.RandomState
        An optional random number generator
    show: bool
        if True the plot will be shown
    **kw:
        Other keywords for the plotter

    Returns
    -------
    the plot object
    """
    import fofx

    plt = fofx.plot_seg(
        seg,
        width=width,
        rng=rng,
        show=False,
        **kw
    )
    add_fofs_to_plot(plt, fofs, rng=rng, show=show, **kw)

    return plt


def add_fofs_to_plot(plt, fofs, rng=None, size=1, minsize=2):
    """
    overplot fofs on the input plot
    """
    import esutil as eu
    import biggles

    if rng is None:
        rng = np.random.RandomState()

    hd = eu.stat.histogram(fofs['fof_id'], min=0, more=True)

    rev = hd['rev']

    low = 50/255
    high = 255/255

    for i in range(hd['hist'].size):
        w = rev[rev[i]:rev[i+1]]

        if w.size >= minsize:

            rgb = rng.uniform(low=low, high=high, size=3)
            color = rgb_to_hex(rgb)

            pts = biggles.Points(
                fofs['x'][w],
                fofs['y'][w],
                type='filled circle',
                size=size,
                color=color,
            )
            plt.add(pts)


def plot_seg(segin, width=1000, rng=None, show=False, **kw):
    """
    plot the seg map with randomized colors for better display
    """
    import images

    if rng is None:
        rng = np.random.RandomState()

    seg = np.transpose(segin)

    cseg = np.zeros((seg.shape[0], seg.shape[1], 3))

    useg = np.unique(seg)[1:]

    low = 50/255
    high = 255/255

    for i, segval in enumerate(useg):

        w = np.where(seg == segval)

        r, g, b = rng.uniform(low=low, high=high, size=3)

        cseg[w[0], w[1], 0] = r
        cseg[w[0], w[1], 1] = g
        cseg[w[0], w[1], 2] = b

    plt = images.view(cseg, show=False, **kw)

    if show:
        srat = seg.shape[1]/seg.shape[0]
        plt.show(width=width, height=width*srat)

    return plt


def rgb_to_hex(rgb):
    """
    convert (r, g, b) tuple to a hex string
    """
    return '#%02x%02x%02x' % rgb
