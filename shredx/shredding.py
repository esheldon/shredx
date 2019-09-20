"""
TODO
    - record error messages
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
               psf_ngauss,
               model='dev',
               fofs=None,
               fill_zero_weight=True,
               min_fofsize=2,
               rng=None,
               get_shredders=False,
               fof_range=None,
               show=False,
               show_full=False,
               **kw):
    """
    Parameters
    ----------
    loader:  shredx.Loader
        The data loader
    psf_ngauss: int
        Number of gaussians to use for the psf fitting
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
    fof_range: 2-element sequence, optional
        Only process the requested range of fofs, formatted like a python slice
        [start, end)
    show: bool
        If True, show some plots for FoF groups
    show_full: bool
        If True, show some plots for the full image
    **kw: extra plotting keywords

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
        loader.view(show=True, rng=rng, **kw)

    cat = loader.cat

    hd = eu.stat.histogram(cat['fof_id'], more=True)

    rev = hd['rev']
    nfofs = hd['hist'].size

    if fof_range is not None:
        assert len(fof_range) == 2
        assert fof_range[0] >= 0
        assert fof_range[1] <= nfofs
        nfofs = fof_range[1] - fof_range[0]
    else:
        fof_range = [0, nfofs]

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
                    psf_ngauss=psf_ngauss,
                    cat=fof_cat,
                    model=model,
                    get_shredder=get_shredders,
                    skip_fit=True,
                    **kw
                )
            else:

                logger.info('-'*70)
                logger.info('processing %d/%d' % (ii+1, nfofs))

                fof_id = cat['fof_id'][ind[0]]
                logger.info(
                    'fof_id: %d fof_size: %d '
                    'ids: %s' % (fof_id, fof_size, str(ind))
                )

                fof_mbobs, fof_seg, fof_cat = loader.get_fof_mbobs(fof_id)
                assert fof_cat.size == ind.size

                kw['seg'] = fof_seg
                res = shred(
                    obs=fof_mbobs,
                    psf_ngauss=psf_ngauss,
                    cat=fof_cat,
                    model=model,
                    rng=rng,
                    show=show,
                    fill_zero_weight=fill_zero_weight,
                    get_shredder=get_shredders,
                    **kw
                )

                if show:
                    if input('hit a key (q to quit): ') == 'q':
                        return

            if get_shredders:
                output, s = res
                shredder_list.append(s)
            else:
                output = res

            if fof_size >= min_fofsize:
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
          psf_ngauss,
          model='dev',
          rng=None,
          show=False,
          fill_zero_weight=True,
          get_shredder=False,
          skip_fit=False,
          **kw):
    """
    deblend objects in the input images

    Parameters
    ----------
    obs: ngmix.MultiBandObsList
        observations with image data
    cat: array with fields
        Array with fields needed for the guesser.  See
        shredder.
    psf_ngauss: int
        Number of gaussians to use for the psf fitting
    model: str
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
    ngauss_per = ngmix.gmix.get_model_ngauss(model)

    if skip_fit:
        res = {'flags': shredder.procflags.NO_ATTEMPT}
        s = None
    else:

        gm_guess = shredder.get_guess(
            cat,
            jacobian=mbobs[0][0].jacobian,
            model=model,
            rng=rng,
        )

        s = shredder.Shredder(
            obs=mbobs,
            psf_ngauss=psf_ngauss,
            fill_zero_weight=fill_zero_weight,
            rng=rng,
        )
        s.shred(gm_guess)

        if show:
            s.plot_comparison(show=True, **kw)

        res = s.get_result()

    tm = time.time() - tm0

    output = _make_output(
        cat=cat,
        psf_ngauss=psf_ngauss,
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

    if 'fof_id' in cat.dtype.names:
        output['fof_id'] = cat['fof_id']

    output['fof_size'] = nobj

    output['flags'] = res['flags']

    # if the user never tried the fit this will not be present
    if 'coadd_result' in res:
        cres = res['coadd_result']
        for n in resfields:
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
