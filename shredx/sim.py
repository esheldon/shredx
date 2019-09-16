import os
import numpy as np
import fitsio
import shredder
from .detect import run_sep_on_mbobs


def get_simulated_files(tmpdir, config=None, rng=None):
    """
    write out simulated data and return paths etc.

    Parameters
    ----------
    tmpdir: str
        The directory in which to write files.  Note the caller is responsible
        for cleaning up, best is using a context manager such as
        tempfile.TemporaryDirectory
    rng: np.random.RandomState
        Optional rng for the shredder simulation.
    config: dict
        Optional configuration parameters for the shredder simulation.

    the psf files returned are actually PSFImageWrapper
    objects, which the loader will use
    """

    sim = shredder.sim.Sim(rng=rng)

    mbobs = sim()

    # run sep to get catalog and seg map, write
    # to files

    image_files = []
    psfs = []
    for band, obslist in enumerate(mbobs):
        obs = obslist[0]

        image_file = os.path.join(tmpdir, 'image-%d.fits' % band)

        hdr = _get_wcs_header()
        with fitsio.FITS(image_file, 'rw') as fits:
            fits.write(obs.image, extname='sci', header=hdr)
            fits.write(obs.weight, extname='wgt')

        image_files.append(image_file)

        psf = PSFImageWrapper(obs.psf.image)
        psfs.append(psf)

    cat, seg = run_sep_on_mbobs(mbobs)
    cat_file = os.path.join(tmpdir, 'cat.fits')
    seg_file = os.path.join(tmpdir, 'seg.fits')

    fitsio.write(cat_file, cat)
    fitsio.write(seg_file, seg)

    return image_files, psfs, seg_file, cat_file


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
