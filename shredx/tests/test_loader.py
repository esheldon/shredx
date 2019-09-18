import logging
import numpy as np
import pytest
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

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        loader.find_fofs()

        if show:
            loader.view(show=True, width=width, rng=rng)

        cat = loader.cat
        fof_ids = np.unique(cat['fof_id'])

        for fof_id in fof_ids:
            logger.info('processing fof: %d' % fof_id)
            ind, = np.where(cat['fof_id'] == fof_id)

            logger.info('ids: %s' % str(ind))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(ind)

            fof_mbobs, fof_seg, fof_cat = loader.get_fof_mbobs(fof_id)

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

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        loader.find_fofs()

        if show:
            loader.view(show=True, width=width, rng=rng)

        cat = loader.cat
        assert np.unique(cat['fof_id']).size == true_nfofs

        fof_ids = np.unique(cat['fof_id'])

        for i, fof_id in enumerate(fof_ids):
            logger.info('processing fof: %d' % fof_id)
            ind, = np.where(cat['fof_id'] == fof_id)

            assert tuple(ind) == true_fof_indices[i]

            logger.info('ids: %s' % str(ind))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(ind)

            tfof_mbobs, tfof_seg, tfof_cat = loader.get_fof_mbobs(fof_id)
            assert fof_cat.size == tfof_cat.size

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
