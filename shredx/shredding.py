import logging
import esutil as eu
import shredder
import fofx

logger = logging.getLogger(__name__)


def shred_fofs(*,
               loader,
               fofs=None,
               model='dev',
               min_fofsize=2,
               rng=None,
               show=False,
               width=1000):
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

    Returns
    -------
    output: list of results
        A list of result dicts from the shredder
    """

    if fofs is not None:
        loader.add_fofs(fofs)
    elif 'fof_id' not in loader.cat.dtype.names:
        loader.find_fofs()

    if show:
        loader.view(show=True, width=width, rng=rng)

    cat = loader.cat

    hd = eu.stat.histogram(cat['fof_id'], more=True)

    rev = hd['rev']

    reslist = []
    for i in range(hd['hist'].size):

        if rev[i] != rev[i+1]:
            w = rev[rev[i]:rev[i+1]]

            numbers = 1+w

            fof_id = cat['fof_id'][w[0]]
            logger.info('fof: %d ids: %s' % (fof_id, str(w)))

            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)

            res = shred(
                mbobs=fof_mbobs,
                cat=fof_cat,
                model=model,
                rng=rng,
            )

            reslist.append(res)

    return reslist


def shred(*, mbobs, cat, model='dev', rng=None):
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
    res: dict
        Result dict from the shredder
    """
    gm_guess = shredder.get_guess(
        cat,
        jacobian=mbobs[0][0].jacobian,
        model=model,
        rng=rng,
    )

    s = shredder.Shredder(mbobs, rng=rng)
    s.shred(gm_guess)

    return s.get_result()
