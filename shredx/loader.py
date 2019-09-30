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
                 mask_ext='msk',
                 pixbuf=10,
                 coord_offset=1,
                 zero_weight_badpix=None,
                 rescale=True,
                 rng=None):
        """
        Parameters
        ----------
        image_files: string or list
            image file path or list of paths.
        psf_files: string or list
            psf file path or list of paths.  For psfs you can
            also send a list of psf objects
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
        zero_weight_badpix: int, optional
            Zero the weight map where these pixels are set in the bitmask
        rescale: bool
            Scale the fluxes by pixel scale squared
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            rng = np.random.RandomState()

        # we load all pixels into the observation, we will usually fill those
        # with zero weight using the model

        self._ignore_zero_weight = False
        self._rescale = rescale

        self._image_ext = image_ext
        self._weight_ext = weight_ext
        self._mask_ext = mask_ext

        self.rng = rng
        self._pixbuf = pixbuf
        self._coord_offset = coord_offset
        self._zero_weight_badpix = zero_weight_badpix
        image_files = _get_file_list(image_files)
        psf_files = _get_file_list(psf_files)
        self.nband = len(image_files)

        npsf = len(psf_files)
        assert self.nband == npsf, \
            ('image and psf lists are '
             'different length: %d %d' % (self.nband, npsf))

        self._load_psfs(psf_files)
        self._load_seg(seg_file)
        self._load_hdus(image_files)
        self._load_cat(cat_file)

    def find_fofs(self):
        """
        find the fof groups and set the fof ids in the catalog
        """
        import fofx

        fofs = fofx.get_fofs(self.seg)
        self.ocat = fofx.add_fofs_to_cat(self.ocat, fofs)
        self.cat = fofx.add_fofs_to_cat(self.cat, fofs)

    def add_fofs(self, fofs):
        """
        add fof information to the catalog
        """

        import fofx
        self.ocat = fofx.add_fofs_to_cat(self.ocat, fofs)
        self.cat = fofx.add_fofs_to_cat(self.cat, fofs)

    def view(self,
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

        if 'fof_id' in self.cat.dtype.names:
            do_fofs = True
        else:
            do_fofs = False

        for im, wt in zip(self.image_hdu_list, self.weight_hdu_list):

            if not isinstance(im, np.ndarray):
                im = im[:, :]
                wt = wt[:, :]

            imlist.append(im)
            wtlist.append(wt)

        if not do_fofs:
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
                self.cat,
                scale=scale,
                minsize=minsize,
                **kw
            )
            segplt = vis.plot_seg_and_fofs(
                seg,
                self.cat,
                minsize=minsize,
                rng=rng,
                **kw
            )

        tab[0, 0] = implt
        tab[0, 1] = segplt

        if show:
            tab.show(width=width, height=int(rat*width))

        return tab

    def get_fof_mbobs(self, fof_id):
        """
        get the mbobs for the requested fof group.  The fof information
        must be present in the catalog, either in the input catalog
        or by running find_fofs()

        Parameters
        ----------
        fof_id: int
            id of the FoF group

        Returns
        -------
        mbobs: ngmix.MultiBandObsList
            observation containing the group
        seg: ngmix.MultiBandObsList
            seg map corresponding to the region in the returned
            images
        cat: array with fields
            Subset of the catalog containing the group, with positions
            adjusted for the new image size
        """

        cat = self.cat

        if 'fof_id' not in cat.dtype.names:
            raise ValueError('no fof_id in catalog, run find_fofs')

        indices, = np.where(cat['fof_id'] == fof_id)
        if indices.size == 0:
            raise ValueError('fof group %d not found' % fof_id)

        return self.get_mbobs(indices)

    def get_mbobs(self, indices):
        """
        Get an ngmix.MultiBandObsList for the region
        encompasing the input object list

        Fill with noise any parts of the image not assigned
        to those objects

        The jacobian from the center of the region is used

        Parameters
        ----------
        indices: array of int
            The 0-offset index values

        Returns
        -------
        mbobs: ngmix.MultiBandObsList
            observation containing the group
        seg: ngmix.MultiBandObsList
            seg map corresponding to the region in the returned
            images
        cat: array with fields
            Subset of the catalog containing the group, with positions
            adjusted for the new image size
        """

        seg = self.seg
        cat = self.cat

        numbers = cat['number'][indices]

        ranges = self._get_image_box(indices)
        minrow, maxrow, mincol, maxcol = ranges

        new_seg = seg[
            minrow:maxrow,
            mincol:maxcol,
        ].copy()

        new_cat = cat[indices].copy()
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
            image, weight, mask = self._get_image_data(band, ranges)
            jacob = self._get_jacobian(band, midrow, midcol)
            psf_obs = self._get_psf_obs(band, midrow, midcol)

            _replace_with_noise(image, weight, wout, self.rng)

            if self._rescale:
                scale = jacob.scale
                image *= 1.0/scale**2
                weight *= scale**4

            # zero the weight map for bad pixels.  Note we are doing this after
            # replacing non-member objects pixels with noise, in case the
            # weight map ends up all zero

            if self._zero_weight_badpix is not None:
                _zero_weight_map_for_badpix(
                    mask, weight, self._zero_weight_badpix,
                )

            obs = ngmix.Observation(
                image,
                weight=weight,
                jacobian=jacob,
                psf=psf_obs,
                ignore_zero_weight=self._ignore_zero_weight,
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

        rrow = round(midrow)
        rcol = round(midcol)

        image = psf.get_rec(rrow, rcol)
        cen = psf.get_center(rrow, rcol)

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
        mask_hdu = self.mask_hdu_list[band]

        image = image_hdu[
            minrow:maxrow,
            mincol:maxcol,
        ]
        weight = weight_hdu[
            minrow:maxrow,
            mincol:maxcol,
        ]
        mask = mask_hdu[
            minrow:maxrow,
            mincol:maxcol,
        ]

        return image, weight, mask

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
        get indices of pixels assigned to objects not in the input list

        Parameters
        ----------
        numbers: array
            Array of sextractor number ids
        seg: array
            The sextractor seg map

        Returns
        -------
        wout: tuple of arrays
            Indices of pixels not assigned to objects not in the input list, as
            returned by the np.where function
        """

        logic = seg > 0

        for i, number in enumerate(numbers):
            tlogic = seg != number
            logic &= tlogic

        wout = np.where(logic)
        return wout

    def _load_cat(self, cat_file):
        """
        load the catalog and make sure it is ordered by number
        """

        logger.info('loading cat file %s' % cat_file)
        ocat = fitsio.read(cat_file, lower=True)

        names = ocat.dtype.names

        if 'number' not in names:
            raise ValueError('catalog must have number field')

        s = ocat['number'].argsort()
        ocat = ocat[s]

        enum = np.arange(1, ocat.size+1)
        assert np.all(ocat['number'] == enum)

        if 'flux' not in names:
            if 'flux_auto' not in names:
                raise ValueError('catalog must have flux or flux_auto')
            add_dt = [('flux', 'f4')]
            ocat = eu.numpy_util.add_fields(ocat, add_dt)
            ocat['flux'] = ocat['flux_auto']
            names = ocat.dtype.names

        req = [
            'xmin', 'xmax', 'ymin', 'ymax',
        ]
        mreq = ['x', 'y', 'x2', 'y2']

        if 'xmin' in names:
            cat = ocat.copy()
        else:
            # we need to copy some data to standard names
            dtype = [
                ('x', 'f8'),
                ('y', 'f8'),
                ('x2', 'f8'),
                ('y2', 'f8'),
                ('xmin', 'i4'),
                ('xmax', 'i4'),
                ('ymin', 'i4'),
                ('ymax', 'i4'),
            ]

            cat = eu.numpy_util.add_fields(ocat, dtype)

            # we prefer win quantities
            wnamer = Namer(back='win_image')
            inamer = Namer(back='_image')

            for name in req:
                if inamer(name) in names:
                    logger.info('%s -> %s' % (inamer(name), name))
                    cat[name] = ocat[inamer(name)]
                else:
                    raise ValueError('no xmin, xmin_image etc. fields found')

            for name in mreq:
                if wnamer(name) in names:
                    logger.info('%s -> %s' % (wnamer(name), name))
                    cat[name] = ocat[wnamer(name)]
                elif inamer(name) in names:
                    logger.info('%s -> %s' % (inamer(name), name))
                    cat[name] = ocat[inamer(name)]
                else:
                    raise ValueError('no x, x_image or xwin_image '
                                     'etc. fields found')

        if self._coord_offset != 0:
            for name in ['x', 'y', 'xmin', 'xmax', 'ymin', 'ymax']:
                cat[name] -= self._coord_offset

        self.ocat = ocat
        self.cat = cat

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
        self.mask_hdu_list = []
        self.wcs_list = []

        for fname in image_files:
            logger.info('loading image file %s' % fname)
            f = fitsio.FITS(fname)

            image_hdu = f[self._image_ext]
            weight_hdu = f[self._weight_ext]
            mask_hdu = f[self._mask_ext]

            self.image_hdu_list.append(image_hdu)
            self.weight_hdu_list.append(weight_hdu)
            self.mask_hdu_list.append(mask_hdu)

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


def _zero_weight_map_for_badpix(mask,
                                weight,
                                badpix):
    """
    zero the weight map for any pixels with the input bitmask
    set
    """

    w = np.where((mask & badpix) != 0)
    if w[0].size > 0:
        logger.info('zeroing weight for %d bad pixels' % w[0].size)
        weight[w] = 0.0


def _get_file_list(image_files):
    if isinstance(image_files, str):
        return [image_files]
    else:
        return image_files


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front == '':
            front = None
        if back == '' or back == 'noshear':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s%s' % (self.front, n)
            if self.back is not None:
                n = '%s%s' % (n, self.back)

        return n
