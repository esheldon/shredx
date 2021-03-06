"""
TODO
    - record versions of packages used
    - record error messages
    - make start/end inclusive to be consistent with fitvd, but also do keep
    the start/num combo which is useful for testing
    - use loader rng if none is sent?
"""
import time
import numpy as np
import logging
import ngmix
import esutil as eu
import shredder

logger = logging.getLogger(__name__)


def shred_fofs(*,
               loader,
               shredconf,
               guess_model='dev',
               fofs=None,
               min_fofsize=2,
               rng=None,
               get_shredders=False,
               fof_range=None,
               # plotting
               show=False,
               showonly=False,
               show_full=False,
               width=1000,
               scale=2,
               title=None):
    """
    Parameters
    ----------
    loader:  shredx.Loader
        The data loader
    shredconf: dict
        A dict passed as options to the shredder. At a minimum this must
        contain 'psf_ngauss'.  See shredder.Shredder for options
    fofs: array with fields
        The fof group data, e.g. returned by fofx.get_fofs
    guess_model: str
        Initial guess for gaussians are laid out with relative size and flux
        based on the model
        'dev', 'exp', 'bdf', 'bd', default 'dev'
    min_fofsize: int
        Minimum fof size to process.  Default is 2, so single
        objects are not processed.
    rng: np.random.RandomState
        Random number generator
    get_shredders: bool
        If true return a list of shredders rather than a list of results
    fof_range: 2-element sequence, optional
        Only process the requested range of fofs [start, end]
    show: bool
        If True, show some plots for FoF groups
    showonly: bool
        If True, just show images don't do fits
    show_full: bool
        If True, show some plots for the full image

    Returns
    -------
    output: list of results
        A list of result dicts from the shredder, or a list of shredders
        if get_shredders == True
    """

    tm0 = time.time()

    if fofs is not None:
        loader.add_fofs(fofs)
    elif 'fof_id' not in loader.cat.dtype.names:
        loader.find_fofs()

    if show_full:
        loader.view(show=True, rng=rng, width=width, scale=scale)

    cat = loader.cat

    hd = eu.stat.histogram(cat['fof_id'], more=True)

    rev = hd['rev']
    nfofs = hd['hist'].size

    if fof_range is not None:
        assert len(fof_range) == 2
        assert fof_range[0] >= 0
        assert fof_range[1] < nfofs
        nfofs = fof_range[1] - fof_range[0] + 1
    else:
        fof_range = [0, nfofs-1]

    reslist = []
    shredder_list = []

    for ii in range(nfofs):
        i = fof_range[0] + ii

        if rev[i] != rev[i+1]:
            ind = rev[rev[i]:rev[i+1]]

            fof_size = ind.size
            if fof_size < min_fofsize:
                fof_cat = cat[ind].copy()
                res = shred(
                    obs=[0]*loader.nband,
                    shredconf=shredconf,
                    cat=fof_cat,
                    guess_model=guess_model,
                    get_shredder=get_shredders,
                    skip_fit=True,
                )
            else:

                logger.info('-'*70)
                logger.info('processing %d:%d' % (i, fof_range[1]))

                fof_id = cat['fof_id'][ind[0]]
                logger.info(
                    'fof_id: %d fof_size: %d '
                    'ids: %s' % (fof_id, fof_size, str(ind))
                )

                fof_mbobs, fof_seg, fof_cat = loader.get_fof_mbobs(fof_id)
                assert fof_cat.size == ind.size

                if showonly:
                    shredder.vis.view_mbobs(
                        fof_mbobs,
                        show=True,
                        width=width,
                        scale=scale,
                        title=title,
                    )
                    res = np.zeros([])
                else:

                    res = shred(
                        obs=fof_mbobs,
                        shredconf=shredconf,
                        cat=fof_cat,
                        guess_model=guess_model,
                        rng=rng,
                        get_shredder=get_shredders,
                        show=show,
                        width=width,
                        scale=scale,
                        title=title,
                        seg=fof_seg,
                    )

                if show or showonly:
                    if input('hit a key (q to quit): ') == 'q':
                        return

            if get_shredders:
                output, s = res
                shredder_list.append(s)
            else:
                output = res

            if fof_size >= min_fofsize and not showonly:
                logger.info('time: %g' % output['time'][0])

            reslist.append(output)

    output = eu.numpy_util.combine_arrlist(reslist)

    tm = time.time() - tm0
    print('total time:', tm)
    print('time per:', tm/nfofs)

    if get_shredders:
        return output, shredder_list
    else:
        return output


def shred(*,
          obs,
          cat,
          shredconf,
          guess_model='dev',
          rng=None,
          get_shredder=False,
          skip_fit=False,
          # plotting
          show=False,
          width=1000,
          scale=2,
          seg=None,  # for plotting
          title=None):
    """
    deblend objects in the input images

    Parameters
    ----------
    obs: ngmix.MultiBandObsList
        observations with image data
    cat: array with fields
        Array with fields needed for the guesser.  See
        shredder.
    shredconf: dict
        A dict passed as options to the shredder. At a minimum this must
    guess_model: str
        Initial guess for gaussians are laid out with relative size and flux
        based on the model
        'dev', 'exp', 'bdf', 'bd', default 'dev'
    rng: np.random.RandomState
        Random number generator
    get_shredder: bool
        If True return output, shredder

    Returns
    -------
    output: array with fields
        The catalog with additional fields added containing
        the result.
    shredder: shredder.Shredder, optional
        The shredder used to deblend. If get_shredder is True, the
        returned value is output, shredder
    """

    mbobs = obs

    tm0 = time.time()

    nband = len(mbobs)
    ngauss_per = ngmix.gmix.get_model_ngauss(guess_model)

    if skip_fit:
        res = {'flags': shredder.procflags.NO_ATTEMPT}
        s = None
    else:

        gm_guess = shredder.get_guess(
            cat,
            jacobian=mbobs[0][0].jacobian,
            model=guess_model,
            rng=rng,
        )

        s = shredder.Shredder(
            obs=mbobs,
            rng=rng,
            **shredconf
        )
        s.shred(gm_guess)

        res = s.get_result()

        if show:
            if res['flags'] == 0:
                s.plot_comparison(
                    show=True,
                    width=width,
                    title=title,
                    scale=scale,
                    seg=seg,
                )
            else:
                shredder.vis.view_mbobs(s.mbobs)

    tm = time.time() - tm0

    output = _make_output(
        cat=cat,
        psf_ngauss=shredconf['psf_ngauss'],
        res=res,
        nband=nband,
        ngauss_per=ngauss_per,
        time=tm,
    )

    if get_shredder:
        return output, s
    else:
        return output


def _make_output(*, cat, psf_ngauss, res, nband, ngauss_per, time):
    """
    combine the input catalog with the rsult

    we will add
    flags

    coadd_flags
    coadd_psf_pars[6]
    pars[ngauss_per*6]

    band_flags[nband]
    band_psf_pars[nband, 6]
    band_pars[nband, ngauss_per*6]
    """

    resfields = ('flags', 'numiter', 'sky')

    nobj = cat.size
    output = _make_output_struct(
        nobj=nobj,
        nband=nband,
        ngauss_per=ngauss_per,
        psf_ngauss=psf_ngauss,
    )

    output['number'] = cat['number']
    output['time'] = time

    output = _add_extra(output, cat)

    if 'fof_id' in cat.dtype.names:
        output['fof_id'] = cat['fof_id']

    output['fof_size'] = nobj

    output['flags'] = res['flags']

    # if the user never tried the fit this will not be present
    if 'coadd_result' in res:
        cres = res['coadd_result']
        for n in resfields:
            if n in cres:
                output['coadd_%s' % n] = cres[n]

        pgmix = res['coadd_psf_gmix']
        ppars = pgmix.get_full_pars()
        output['coadd_psf_T'] = pgmix.get_T()
        for i in range(nobj):
            output['coadd_psf_pars'][i, :] = ppars

        gmix = res['coadd_gmix']
        if gmix is not None:
            pars = gmix.get_full_pars()
            output['coadd_pars'] = pars.reshape(output['coadd_pars'].shape)

            for i in range(nobj):
                ipars = output['coadd_pars'][i]
                igm = ngmix.GMix(pars=ipars)
                output['coadd_T'][i] = igm.get_T()
                output['coadd_flux'][i] = igm.get_flux()

        if 'band_results' in res:
            for band in range(nband):
                bres = res['band_results'][band]
                for n in resfields:
                    if n in bres:
                        output['band_%s' % n][:, band] = bres[n]

                pgmix = res['band_psf_gmix'][band]
                bppars = pgmix.get_full_pars()
                output['band_psf_T'][:, band] = pgmix.get_T()
                for i in range(nobj):
                    output['band_psf_pars'][i, :, band] = bppars

                bgmix = res['band_gmix'][band]
                if bgmix is not None:
                    bpars = bgmix.get_full_pars()
                    bpars = bpars.reshape(nobj, 6*ngauss_per)
                    output['band_pars'][:, :, band] = bpars

                    for i in range(nobj):
                        ipars = output['band_pars'][i, :, band]
                        igm = ngmix.GMix(pars=ipars)
                        output['band_flux'][i, band] = igm.get_flux()

    return output


def _add_extra(output, cat):
    """
    add optional extra fields from the catalog
    """

    names = cat.dtype.names
    if 'ra' in names:
        output['ra'] = cat['ra']
        output['dec'] = cat['dec']

    elif 'alphawin_j2000' in names:
        output['ra'] = cat['alphawin_j2000']
        output['dec'] = cat['deltawin_j2000']

    if 'id' in names:
        output = add_ids(output, cat['id'])

    return output


def add_ids(cat, ids):
    """
    add id information to the output

    may generate a new catalog if 'id' is not present
    """

    logger.info('adding ids')

    mids, mcat = match_ids(cat, ids)

    if 'id' not in cat.dtype.names:
        add_dt = [('id', 'i8')]
        cat = eu.numpy_util.add_fields(cat, add_dt)

    cat['id'][mcat] = ids['id'][mids]

    return cat


def match_ids(cat, ids):
    """
    match the catalog to the ids catalog
    """
    mids, mcat = eu.numpy_util.match(ids['number'], cat['number'])

    if mcat.size != cat.size:
        raise ValueError('not all numbers matched to id catalog')

    return mids, mcat


def _make_output_struct(*,
                        nobj,
                        nband,
                        ngauss_per,
                        psf_ngauss):
    """
    make an output structure
    """
    from shredder import procflags

    dt = [
        ('number', 'i4'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('fof_id', 'i4'),
        ('fof_size', 'i4'),
        ('flags', 'i4'),
        ('coadd_flags', 'i4'),
        ('coadd_numiter', 'i4'),
        ('coadd_sky', 'f4'),
        ('coadd_psf_pars', 'f8', 6*psf_ngauss),
        ('coadd_psf_T', 'f8'),
        ('coadd_pars', 'f8', 6*ngauss_per),
        ('coadd_T', 'f8'),
        ('coadd_flux', 'f8'),

        ('band_flags', 'i4', nband),
        ('band_numiter', 'i4', nband),
        ('band_sky', 'f4', nband),
        ('band_psf_pars', 'f8', (6*psf_ngauss, nband)),
        ('band_psf_T', 'f8', nband),
        ('band_pars', 'f8', (6*ngauss_per, nband)),
        ('band_flux', 'f8', nband),

        ('time', 'f4'),
    ]

    st = np.zeros(nobj, dtype=dt)

    for n in st.dtype.names:
        if 'flags' in n:
            st[n] = procflags.NO_ATTEMPT
        else:
            st[n] = -9999

    return st
