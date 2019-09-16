import logging
import numpy as np
import pytest
import shredx
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('seed', [6419, 491])
def test_loader_smoke(seed):
    """
    test that the loader doesn't crash
    """

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:
        image_files, psfs, seg_file, cat_file = \
            shredx.sim.get_simulated_files(tmpdir, rng=rng)

        loader = shredx.Loader(
            image_files=image_files,
            psf_files=psfs,
            seg_file=seg_file,
            cat_file=cat_file,
            coord_offset=0,
            rng=rng,
        )

        # or whatever
        numbers = [1, 2, 3]

        tmbobs, tseg, tcat = loader.get_mbobs(numbers)  # noqa


@pytest.mark.parametrize('seed', [2510, 21064])
def test_loader(seed, show=False):
    """
    test we can run the fof finder and extract the group
    """

    import fofx

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:
        image_files, psfs, seg_file, cat_file = \
            shredx.sim.get_simulated_files(tmpdir, rng=rng)

        loader = shredx.Loader(
            image_files=image_files,
            psf_files=psfs,
            seg_file=seg_file,
            cat_file=cat_file,
            coord_offset=0,
            rng=rng,
        )

        if show:
            loader.view(show=True, title='full image')

        fofs = fofx.get_fofs(loader.seg)

        fof_ids = np.unique(fofs['fof_id'])

        for fof_id in fof_ids:
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(fofs['fof_id'] == fof_id)
            numbers = 1+w

            logger.info('ids: %s numbers: %s' % (str(w), str(numbers)))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)  # noqa

            if show:
                import shredder
                title = 'FoF %d' % fof_id
                shredder.vis.view_mbobs(
                    fof_mbobs,
                    scale=2,
                    show=True,
                    title=title,
                )
