import logging
import numpy as np
import pytest
import fofx
import shredder
import shredx
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('seed', [2510, 21064])
def test_shredding_smoke(seed, width=1000, show=False):
    """
    test we can run the fof finder and extract the group
    """

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)

        fofs = fofx.get_fofs(loader.seg)
        cat = fofx.add_fofs_to_cat(loader.cat, fofs)

        if show:
            loader.view(fofs=cat, show=True, width=width, rng=rng)

        fof_ids = np.unique(cat['fof_id'])

        for fof_id in fof_ids:
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(cat['fof_id'] == fof_id)
            numbers = 1+w

            logger.info('ids: %s numbers: %s' % (str(w), str(numbers)))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)
            logger.debug(fof_mbobs[0][0].jacobian)

            fof_cat = fofx.add_fofs_to_cat(fof_cat, fofs)

            if show:
                shredx.vis.plot_mbobs_and_fofs(
                    fof_mbobs,
                    fof_cat,
                    show=True,
                    width=width,
                    rng=rng,
                )

            gm_guess = shredder.get_guess_from_cat(
                fof_cat,
                # pixel_scale=fof_mbobs[0][0].jacobian.scale,
                jacobian=fof_mbobs[0][0].jacobian,
                model='dev',
                rng=rng,
            )

            s = shredder.Shredder(fof_mbobs, rng=rng)
            s.shred(gm_guess)

            res = s.get_result()
            assert res['flags'] == 0

            if show:
                s.plot_comparison(show=True)


def test_shredding(width=1000, show=False):
    """
    test we can run the fof finder and extract the group
    """

    # pick one we know does ok
    seed = 25

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)

        fofs = fofx.get_fofs(loader.seg)
        cat = fofx.add_fofs_to_cat(loader.cat, fofs)

        if show:
            loader.view(fofs=cat, show=True, width=width, rng=rng)

        fof_ids = np.unique(cat['fof_id'])

        for fof_id in fof_ids:
            logger.info('processing fof: %d' % fof_id)
            w, = np.where(cat['fof_id'] == fof_id)
            numbers = 1+w

            logger.info('ids: %s numbers: %s' % (str(w), str(numbers)))
            fof_mbobs, fof_seg, fof_cat = loader.get_mbobs(numbers)
            logger.debug(fof_mbobs[0][0].jacobian)

            fof_cat = fofx.add_fofs_to_cat(fof_cat, fofs)

            if show:
                shredx.vis.plot_mbobs_and_fofs(
                    fof_mbobs,
                    fof_cat,
                    show=True,
                    width=width,
                    rng=rng,
                )

            gm_guess = shredder.get_guess_from_cat(
                fof_cat,
                # pixel_scale=fof_mbobs[0][0].jacobian.scale,
                jacobian=fof_mbobs[0][0].jacobian,
                model='dev',
                rng=rng,
            )

            s = shredder.Shredder(fof_mbobs, rng=rng)
            s.shred(gm_guess)

            res = s.get_result()
            assert res['flags'] == 0

            if show:
                s.plot_comparison(show=True)

            models = s.get_model_images()

            chi2 = 0.0
            dof = 0
            for band, model in enumerate(models):
                image = fof_mbobs[band][0].image
                dof += image.size

                weight = fof_mbobs[band][0].weight
                diffim = image - model
                chi2 += (diffim**2 * weight).sum()

            dof = dof - 3
            chi2per = chi2/dof

            assert chi2per < 1.05
