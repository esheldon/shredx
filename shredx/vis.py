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

    low = 0
    high = 125

    if rng is None:
        seed = hd['hist'].size
        rng = np.random.RandomState(seed)

    for i in range(hd['hist'].size):
        w = rev[rev[i]:rev[i+1]]

        if w.size >= minsize:

            rgb = rng.uniform(low=low, high=high, size=3).astype('i4')
            rgb = tuple(rgb)
            rgb2 = (255-rgb[0], 255-rgb[1], 255-rgb[2])
            color = rgb_to_hex(rgb)
            color2 = rgb_to_hex(rgb2)
            colors = (color, color2)
            # symbol = _get_random_symbol(rng)
            symbols = _get_random_symbols(rng)

            """
            print('colors:', colors)
            print('symbols:', symbols)

            pts1 = biggles.Points(
                fofs['x'][w],
                fofs['y'][w],
                type=symbols[0],
                color=color,
                size=size,
            )
            pts2 = biggles.Points(
                fofs['x'][w],
                fofs['y'][w],
                type=symbols[1],
                color=color2,
                size=size,
            )

            plt.add(pts1, pts2)
            """

            for color, symbol in zip(colors, symbols):
                print("'%s' '%s'" % (symbol, color))
                pts = biggles.Points(
                    fofs['x'][w],
                    fofs['y'][w],
                    type=symbol,
                    color=color,
                    size=size,
                )
                plt.add(pts)


def rgb_to_hex(rgb):
    """
    convert (r, g, b) tuple to a hex string
    """
    return '#%02x%02x%02x' % rgb


def _get_random_symbol(rng):
    i = rng.randint(0, len(_SYMBOLS)-1)
    return _SYMBOLS[i]


def _get_random_symbols(rng):
    i = rng.randint(0, len(_SYMBOL_PAIRS)-1)
    return _SYMBOL_PAIRS[i]


_SYMBOL_PAIRS = (
    ('filled circle', 'circle'),
    ('filled square', 'square'),
    ('filled triangle', 'triangle'),
    ('filled octagon', 'octagon'),
    ('filled diamond', 'diamond'),
    ('filled inverted triangle', 'inverted triangle'),
)

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
