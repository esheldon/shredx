import logging
import numpy as np
import pytest
import fofx
import shredx
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('seed', [2510, 21064])
def test_loader_smoke(seed, width=1000, show=False):
    """
    test we can run the fof finder and extract the group
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


        fofs = fofx.get_fofs(loader.seg)
        cat = fofx.add_fofs_to_cat(loader.cat,fofs)

        if show:
            loader.view(fofs=cat, show=True, width=width, rng=rng)

        fof_ids = np.unique(cat['fof_id'])

        for fof_id in fof_ids:
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(cat['fof_id'] == fof_id)
            numbers = 1+w

            logger.info('ids: %s numbers: %s' % (str(w), str(numbers)))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)

            if show:
                import shredder
                title = 'FoF %d' % fof_id
                shredder.vis.view_mbobs(
                    fof_mbobs,
                    scale=2,
                    show=True,
                    width=width,
                    title=title,
                )


def test_loader(show=False, width=1000):
    """
    test we can run the fof finder and extract the group
    """

    seed = 2510

    true_nfofs = 2
    true_fof_indices = (
        (0, 1, 2),
        (3, 4),
    )

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

        fofs = fofx.get_fofs(loader.seg)
        cat = fofx.add_fofs_to_cat(loader.cat,fofs)

        if show:
            loader.view(fofs=cat, show=True, width=width, rng=rng)

        assert np.unique(fofs['fof_id']).size == true_nfofs

        fof_ids = np.unique(fofs['fof_id'])

        for i, fof_id in enumerate(fof_ids):
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(fofs['fof_id'] == fof_id)
            numbers = 1+w

            assert tuple(w) == true_fof_indices[i]

            logger.info('ids: %s numbers: %s' % (str(w), str(numbers)))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)

            if show:
                import shredder
                title = 'FoF %d' % fof_id
                shredder.vis.view_mbobs(
                    fof_mbobs,
                    scale=2,
                    show=True,
                    width=width,
                    title=title,
                )
