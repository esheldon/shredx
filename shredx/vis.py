import numpy as np

def plot_image_and_fofs(imlist, wtlist, fofs,
                        scale=2,
                        minsize=2, width=1000,
                        rng=None, show=False, **kw):
    """
    plot the fof groups over the rgb image

    Parameters
    ----------
    imlist: list
        list of arrays
    wtlist: list
        list of arrays
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
    import shredder
    import fofx

    plt = shredder.vis.view_rgb(
        imlist,
        wtlist,
        scale=scale,
        **kw
    )

    add_fofs_to_plot(plt, fofs, minsize=minsize, rng=rng, **kw)

    if show:
        tim = imlist[0]
        srat = tim.shape[1]/tim.shape[0]
        plt.show(width=width, height=int(width*srat))

    return plt


def plot_seg_and_fofs(seg, fofs,
                      minsize=2, width=1000,
                      rng=None, show=False, **kw):
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
    add_fofs_to_plot(plt, fofs, minsize=minsize, **kw)

    if show:
        srat = seg.shape[1]/seg.shape[0]
        plt.show(width=width, height=int(width*srat))

    return plt


def add_fofs_to_plot(plt, fofs, rng=None, size=1, minsize=2):
    """
    overplot fofs on the input plot
    """
    import esutil as eu
    import biggles


    hd = eu.stat.histogram(fofs['fof_id'], min=0, more=True)

    rev = hd['rev']

    low = 20
    high = 200

    if rng is None:
        seed = hd['hist'].size
        rng = np.random.RandomState(seed)

    for i in range(hd['hist'].size):
        w = rev[rev[i]:rev[i+1]]

        if w.size >= minsize:

            rgb = rng.uniform(low=low, high=high, size=3).astype('i4')
            color = rgb_to_hex(tuple(rgb))
            symbol = _get_random_symbol(rng)

            pts = biggles.Points(
                fofs['x'][w],
                fofs['y'][w],
                # type='filled circle',
                type=symbol,
                size=size,
                color=color,
            )
            plt.add(pts)


def rgb_to_hex(rgb):
    """
    convert (r, g, b) tuple to a hex string
    """
    print('rgb:', rgb)
    return '#%02x%02x%02x' % rgb


def _get_random_symbol(rng):
    i = rng.randint(0, len(_SYMBOLS)-1)
    return _SYMBOLS[i]


_SYMBOLS = [
    "filled circle",
    "dot", "filled square",
    "plus", "filled triangle",
    "asterisk", "filled diamond",
    "circle", "filled inverted triangle",
    "cross", "filled fancy square",
    "square", "filled fancy diamond",
    "triangle", "half filled circle",
    "diamond", "half filled square",
    "star", "half filled triangle",
    "inverted triangle", "half filled diamond",
    "starburst", "half filled inverted triangle",
    "fancy plus", "half filled fancy square",
    "fancy cross", "half filled fancy diamond",
    "fancy square", "octagon",
    "fancy diamond", "filled octagon",
]
