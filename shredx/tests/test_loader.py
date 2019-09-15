"""
TODO
    - example psfex file
    - example g,r,i,z image files with weight maps
    - example seg file
    - example cat file
    - we can generate these using the simulation in
      shredder, with a fake psfex file from anywhere.
      will need to fake the wcs header
"""

import os
import numpy as np
import pytest
import fitsio
import shredder
import shredx
from tempfile import TemporaryDirectory


class PSFImageWrapper(object):
    """
    provide a PSFEx like interface to an image
    """
    def __init__(self, image):
        self._image = image
        self._cen = (np.array(image.shape)-1)/2

    def get_rec(self, *args):
        """
        just return a copy of the image
        """
        return self._image.copy()

    def get_center(self, *args):
        """
        just return the canonical center
        """
        return self._cen.copy()


def _get_wcs_header():
    cards = [
        "NAXIS   =                    2 / 2-dimensional binary table",
        "NAXIS1  =                 2048 / length of data axix 1",
        "NAXIS2  =                 4096 / length of data axix 2",
        "CRPIX1  =                6660. / Reference pixel on this axis",
        "CRPIX2  =               -4341. / Reference pixel on this axis",
        "CCDBIN1 =                    1 / Pixel binning, axis 1",
        "CCDBIN2 =                    1 / Pixel binning, axis 2",
        "CUNIT1  = 'deg     '           / Axis unit",
        "CD1_1   =   1.013141083499E-08 / Linear projection matrix",
        "CD1_2   =   7.285974030215E-05 / Linear projection matrix",
        "CUNIT2  = 'deg     '           / Axis unit",
        "CD2_1   =  -7.285598092499E-05 / Linear projection matrix",
        "CD2_2   =   1.572634539309E-08 / Linear projection matrix",
        "CTYPE2  = 'DEC--TPV'           / WCS projection type for this axis",
        "CTYPE1  = 'RA---TPV'           / WCS projection type for this axis",
        "CRVAL1  =       70.12751953531 / World coordinate on this axis",
        "CRVAL2  =      -54.55415863979 / World coordinate on this axis",
    ]
    return fitsio.FITSHDR(cards)


def _run_sep_on_mbobs(mbobs):
    """
    run sep and get a cat and seg map
    """
    import shredder
    import sxdes

    coadd_obs = shredder.coadding.make_coadd_obs(mbobs)

    noise = np.sqrt(1.0/coadd_obs.weight[0, 0])
    cat, seg = sxdes.run_sep(coadd_obs.image, noise)
    return cat, seg


def _write_simulated_files(tmpdir, seed):
    """
    write out simulated data

    the psf files returned are actually PSFImageWrapper
    objects, which the loader will use
    """
    rng = np.random.RandomState(seed)
    sim = shredder.sim.Sim(rng=rng)

    mbobs = sim()

    # run sep to get catalog and seg map, write
    # to files

    image_files = []
    psfs = []
    for band, obslist in enumerate(mbobs):
        obs = obslist[0]

        image_file = os.path.join(tmpdir, 'image-%d.fits')

        hdr = _get_wcs_header()
        with fitsio.FITS(image_file, 'rw') as fits:
            fits.write(obs.image, extname='sci', header=hdr)
            fits.write(obs.weight, extname='wgt')

        image_files.append(image_file)

        psf = PSFImageWrapper(obs.psf.image)
        psfs.append(psf)

    cat, seg = _run_sep_on_mbobs(mbobs)
    cat_file = os.path.join(tmpdir, 'cat-%d.fits')
    seg_file = os.path.join(tmpdir, 'seg-%d.fits')

    fitsio.write(cat_file, cat)
    fitsio.write(seg_file, seg)

    return image_files, psfs, seg_file, cat_file


@pytest.mark.parametrize('seed', [6419, 491])
def test_loader_smoke(seed):
    """
    test that the loader doesn't crash
    """
    with TemporaryDirectory() as tmpdir:
        image_files, psfs, seg_file, cat_file = \
            _write_simulated_files(tmpdir, seed)

        loader = shredx.Loader(
            image_files=image_files,
            psf_files=psfs,
            seg_file=seg_file,
            cat_file=cat_file,
            coord_offset=0,
        )

        # or whatever
        numbers = [1, 2, 3]

        mbobs = loader.get_mbobs(numbers)  # noqa
