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

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        loader.find_fofs()

        if show:
            loader.view(show=True, width=width, rng=rng)

        cat = loader.cat
        assert np.unique(cat['fof_id']).size == true_nfofs

        fof_ids = np.unique(cat['fof_id'])

        for i, fof_id in enumerate(fof_ids):
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(cat['fof_id'] == fof_id)
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
