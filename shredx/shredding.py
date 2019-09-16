import logging
import esutil as eu
import shredder

logger = logging.getLogger(__name__)


def shred_fofs(*,
               loader,
               fofs=None,
               model='dev',
               min_fofsize=2,
               rng=None,
               get_shredders=False,
               show=False,
               **kw):
    """
    Parameters
    ----------
    loader:  shredx.Loader
        The data loader
    fofs: array with fields
        The fof group data, e.g. returned by fofx.get_fofs
    model: str
        'dev', 'exp', 'bdf', 'bd', default 'dev'
    min_fofsize: int
        Minimum fof size to process.  Default is 2, so single
        objects are not processed.
    rng: np.random.RandomState
        Random number generator
    get_shredders: bool
        If true return a list of shredders rather than a list of results
    show: bool
        If True, show some plots
    **kw: extra plotting keywords

    Returns
    -------
    output: list of results
        A list of result dicts from the shredder, or a list of shredders
        if get_shredders == True
    """

    if fofs is not None:
        loader.add_fofs(fofs)
    elif 'fof_id' not in loader.cat.dtype.names:
        loader.find_fofs()

    if show:
        loader.view(show=True, rng=rng, **kw)

    cat = loader.cat

    hd = eu.stat.histogram(cat['fof_id'], more=True)

    rev = hd['rev']
    num = hd['hist'].size

    reslist = []
    for i in range(num):

        if rev[i] != rev[i+1]:
            w = rev[rev[i]:rev[i+1]]

            if w.size < min_fofsize:
                # TODO save some information for these anyway?
                continue

            numbers = 1+w

            fof_id = cat['fof_id'][w[0]]
            logger.info('fof: %d ids: %s' % (fof_id, str(w)))

            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)

            s = shred(
                mbobs=fof_mbobs,
                cat=fof_cat,
                model=model,
                rng=rng,
                show=show,
                **kw
            )

            if get_shredders:
                reslist.append(s)
            else:
                reslist.append(s.get_result())

        if show and i < num-1:
            if input('hit a key (q to quit): ') == 'q':
                return

    return reslist


def shred(*, mbobs, cat, model='dev', rng=None, show=False, **kw):
    """
    deblend objects in the input images

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        observations with image data
    cat: array with fields
        Array with fields needed for the guesser.  See
        shredder.
    model: str
        'dev', 'exp', 'bdf', 'bd', default 'dev'
    rng: np.random.RandomState
        Random number generator

    Returns
    -------
    shredder: shredder.Shredder
        The shredder used to deblend
    """

    gm_guess = shredder.get_guess(
        cat,
        jacobian=mbobs[0][0].jacobian,
        model=model,
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng)
    s.shred(gm_guess)

    if show:
        s.plot_comparison(show=True, **kw)

    return s
