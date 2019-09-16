import logging
import numpy as np
import fitsio
import esutil as eu
import ngmix
from . import vis

logger = logging.getLogger(__name__)


class Loader(object):
    """
    Interface to load data from files
    """
    def __init__(self, *,
                 image_files,
                 psf_files,
                 seg_file,
                 cat_file,
                 image_ext='sci',
                 weight_ext='wgt',
                 pixbuf=10,
                 coord_offset=1,
                 rng=None):
        """
        Parameters
        ----------
        image_files: string or list
            image file path or list of paths.
        psf_files: string or list
            psf file path or list of paths.
        seg_file: str
            Path to seg map file
        cat_file: str
            Path to the sextractor catalog file
        image_ext: str
            Extension name or number for image data, default
            'sci' which is the DES convention
        weight_ext: str
            Extension name or number for weight map, default
            'wgt' which is the DES convention
        pixbuf: int
            Buffer to pad around each image extracted around
            a set of objects
        coord_offset: int
            sextractor position offsets, default 1 which is the sextractor
            convention.  Note if you used sep for object extraction, the offset
            is zero
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            rng = np.random.RandomState()

        self._image_ext = image_ext
        self._weight_ext = weight_ext

        self._rng = rng
        self._pixbuf = pixbuf
        self._coord_offset = coord_offset
        image_files = _get_file_list(image_files)
        psf_files = _get_file_list(psf_files)
        self.nband = len(image_files)

        npsf = len(psf_files)
        assert self.nband == npsf, \
            ('image and psf lists are '
             'different length: %d %d' % (self.nband, npsf))

        self._load_psfs(psf_files)
        self._load_seg(seg_file)
        self._load_cat(cat_file)
        self._load_hdus(image_files)

    def view(self,
             fofs=None,
             scale=2,
             show=False,
             width=1000,
             minsize=2,
             rng=None,
             **kw):
        """
        make a plot of the data
        """
        import shredder
        import biggles
        import fofx

        if rng is None:
            rng = np.random.RandomState()

        seg = self.seg
        rat = seg.shape[1]/(2*seg.shape[0])

        tab = biggles.Table(1, 2, aspect_ratio=rat)

        imlist = []
        wtlist = []

        for im, wt in zip(self.image_hdu_list, self.weight_hdu_list):

            if not isinstance(im, np.ndarray):
                im = im[:, :]
                wt = wt[:, :]

            imlist.append(im)
            wtlist.append(wt)

        if fofs is None:
            implt = shredder.vis.view_rgb(
                imlist,
                wtlist,
                scale=scale,
                **kw
            )
            segplt = fofx.vis.plot_seg(
                seg,
                rng=rng,
                **kw
            )

        else:

            implt = vis.plot_image_and_fofs(
                imlist,
                wtlist,
                fofs,
                scale=scale,
                minsize=minsize,
                **kw
            )
            segplt = vis.plot_seg_and_fofs(
                seg,
                fofs,
                minsize=minsize,
                rng=rng,
                **kw
            )

        tab[0, 0] = implt
        tab[0, 1] = segplt

        if show:
            tab.show(width=width, height=int(rat*width))

        return tab

    def get_mbobs(self, numbers):
        """
        Get an ngmix.MultiBandObsList for the region
        encompasing the input object list

        Fill with noise any parts of the image not assigned
        to those objects

        The jacobian from the center of the region is used
        """

        numbers = np.array(numbers, ndmin=1, copy=False)
        ind = numbers - 1

        seg = self.seg
        cat = self.cat

        ranges = self._get_image_box(ind)
        minrow, maxrow, mincol, maxcol = ranges

        new_seg = seg[
            minrow:maxrow,
            mincol:maxcol,
        ].copy()

        new_cat = cat[ind].copy()
        new_cat['y'] -= minrow
        new_cat['x'] -= mincol

        wout = self._get_outside_pixels(numbers, new_seg)
        new_seg[wout] = 0

        mbobs = self._extract_mbobs(ranges, wout)

        return mbobs, new_seg, new_cat

    def _extract_mbobs(self, ranges, wout):
        """
        extract an ngmix.MultiBandObsList for the given region, filling noise
        in the regions not assigned to the requested objects

        Parameters
        ----------
        ranges: tuple
            (minrow, maxrow, mincol, maxcol)
        wout: tuple of arrays
            (wrows, wcols) indices for areas not assigned to
            the objects of interest

        Returns
        -------
        mbobs: ngmix.MultiBandObsList
        """

        minrow, maxrow, mincol, maxcol = ranges

        midrow = (minrow + maxrow)/2
        midcol = (mincol + maxcol)/2

        mbobs = ngmix.MultiBandObsList()

        for band in range(self.nband):
            image, weight = self._get_image_data(band, ranges)
            jacob = self._get_jacobian(band, midrow, midcol)
            psf_obs = self._get_psf_obs(band, midrow, midcol)

            _replace_with_noise(image, weight, wout, self._rng)

            obs = ngmix.Observation(
                image,
                weight=weight,
                jacobian=jacob,
                psf=psf_obs,
            )

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def _get_psf_obs(self, band, midrow, midcol):
        """
        get the image and wieght for the specified box

        Parameters
        ----------
        band: int
            band to load
        ranges: tuple
            (minrow, maxrow, mincol, maxcol)

        Returns
        -------
        image, weight
        """

        psf = self.psf_list[band]

        image = psf.get_rec(midrow, midcol)
        cen = psf.get_center()

        weight = image*0 + 1.0/0.001**2

        jacob = self._get_jacobian(band, midrow, midcol)

        jacob.set_cen(row=cen[0], col=cen[1])

        return ngmix.Observation(
            image,
            weight=weight,
            jacobian=jacob,
        )

    def _get_image_data(self, band, ranges):
        """
        get the image and wieght for the specified box

        Parameters
        ----------
        band: int
            band to load
        ranges: tuple
            (minrow, maxrow, mincol, maxcol)

        Returns
        -------
        image, weight
        """
        minrow, maxrow, mincol, maxcol = ranges

        image_hdu = self.image_hdu_list[band]
        weight_hdu = self.weight_hdu_list[band]

        image = image_hdu[
            minrow:maxrow,
            mincol:maxcol,
        ]
        weight = weight_hdu[
            minrow:maxrow,
            mincol:maxcol,
        ]

        return image, weight

    def _get_jacobian(self, band, midrow, midcol):
        """
        get the ngmix jacobian at the specified location

        "center" of the jacobian will be 0, 0 which is most useful when the
        objects can be anywhere in the image
        """

        wcs = self.wcs_list[band]

        jdata = wcs.get_jacobian(
            x=midcol+self._coord_offset,
            y=midrow+self._coord_offset,
        )

        jacob = ngmix.Jacobian(
            row=0,
            col=0,
            dudcol=jdata[0],
            dudrow=jdata[1],
            dvdcol=jdata[2],
            dvdrow=jdata[3],
        )
        return jacob

    def _get_image_box(self, ind):
        """
        get the region containing all the objects

        Parameters
        ----------
        ind: array
            Indices of the objects of interest

        Returns
        -------
        ranges: tuple
            (minrow, maxrow, mincol, maxcol)
        """
        seg = self.seg
        cat = self.cat
        pixbuf = self._pixbuf

        mincol = cat['xmin'][ind].min() - pixbuf
        maxcol = cat['xmax'][ind].max() + pixbuf + 1
        minrow = cat['ymin'][ind].min() - pixbuf
        maxrow = cat['ymax'][ind].max() + pixbuf + 1

        if minrow < 0:
            minrow = 0
        if maxrow > seg.shape[0]:
            maxrow = seg.shape[0]

        if mincol < 0:
            mincol = 0
        if maxcol > seg.shape[1]:
            maxcol = seg.shape[1]

        return minrow, maxrow, mincol, maxcol

    def _get_outside_pixels(self, numbers, seg):
        """
        get indices of pixels not included in the
        seg maps associated with the input numbers

        Parameters
        ----------
        numbers: array
            Array of sextractor number ids
        seg: array
            The sextractor seg map

        Returns
        -------
        wout: tuple of arrays
            Indices of pixels not assigned to the specified objects, as
            returned by the np.where function
        """

        for i, number in enumerate(numbers):
            tlogic = seg != number
            if i == 0:
                logic = tlogic
            else:
                logic &= tlogic

        wout = np.where(logic)
        return wout

    def _load_cat(self, cat_file):
        """
        load the catalog and make sure it is ordered by number
        """

        logger.info('loading cat file %s' % cat_file)
        cat = fitsio.read(cat_file, lower=True)

        req = ['number', 'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax']
        for n in req:
            if n not in cat.dtype.names:
                raise ValueError('catalog must have field %s' % n)

        s = cat['number'].argsort()
        self.cat = cat[s]

        if self._coord_offset != 0:
            for n in ['x', 'y', 'xmin', 'xmax', 'ymin', 'ymax']:
                self.cat[n] -= self._coord_offset

        enum = np.arange(1, cat.size+1)
        assert np.all(cat['number'] == enum)

    def _load_seg(self, seg_file):
        """
        load the seg map into memory
        """
        logger.info('loading seg file %s' % seg_file)
        self.seg = fitsio.read(seg_file)

    def _load_hdus(self, image_files):
        """
        load image and weight hdus
        """

        self.image_hdu_list = []
        self.weight_hdu_list = []
        self.wcs_list = []

        for fname in image_files:
            logger.info('loading image file %s' % fname)
            f = fitsio.FITS(fname)

            image_hdu = f[self._image_ext]
            weight_hdu = f[self._weight_ext]

            self.image_hdu_list.append(image_hdu)
            self.weight_hdu_list.append(weight_hdu)

            header = image_hdu.read_header()
            wcs = eu.wcsutil.WCS(header)
            self.wcs_list.append(wcs)

    def _load_psfs(self, psf_files):
        """
        load the psfs
        """
        import psfex

        self.psf_list = []
        for i, psf_file in enumerate(psf_files):
            if isinstance(psf_file, str):
                logger.info('loading psfex file %s' % psf_file)
                p = psfex.PSFEx(psf_file)
            else:
                if i == 0:
                    logger.info('got non-string psf input, assuming '
                                'are psfs with correct interface')
                p = psf_file

            self.psf_list.append(p)


def _replace_with_noise(image, weight, indices, rng):
    """
    replace pixels with noise
    """
    noise = np.sqrt(1.0/weight.max())
    noise_image = rng.normal(
        scale=noise,
        size=image.shape
    )
    image[indices] = noise_image[indices]


def _get_file_list(image_files):
    if isinstance(image_files, str):
        return [image_files]
    else:
        return image_files
