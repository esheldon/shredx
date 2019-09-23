import logging
import numpy as np
import pytest
import shredx
import shredder
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('seed', [2510, 21064])
def test_shredding_smoke(seed, show=False, **kw):
    """
    test we can run the fof finder and extract the group
    """

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    conf = {'psf_ngauss': 2}

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        output = shredx.shred_fofs(
            loader=loader,
            shredconf=conf,
            show=show, rng=rng,
            **kw
        )

    return output


def test_shredding(show=False, **kw):
    """
    test we can run the fof finder and extract the group
    with decent fidelity

    choose a seed for which we know we get a decent answer

    We don't use shred_fofs because we want to do individual
    checks
    """

    # pick one we know does ok
    seed = 25

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    conf = {'psf_ngauss': 2}

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        outputs, shredders = shredx.shred_fofs(
            loader=loader,
            shredconf=conf,
            show=show,
            get_shredders=True,
            rng=rng,
            **kw
        )

        assert outputs.size == loader.cat.size

        for s in shredders:

            if s is None:
                continue

            res = s.get_result()
            assert res['flags'] == 0

            models = s.get_model_images()

            chi2 = 0.0
            dof = 0
            for band, model in enumerate(models):
                image = s.mbobs[band][0].image
                dof += image.size

                weight = s.mbobs[band][0].weight
                diffim = image - model
                chi2 += (diffim**2 * weight).sum()

            dof = dof - 3
            chi2per = chi2/dof

            assert chi2per < 1.05


@pytest.mark.parametrize('seed', [995, 7317])
def test_shredding_bad_columns(seed, show=False, **kw):
    """
    test with bad columns
    """

    shredx.setup_logging('debug')

    logger.info('seed: %d' % seed)
    rng = np.random.RandomState(seed)

    conf = {'psf_ngauss': 2}

    with TemporaryDirectory() as tmpdir:

        config = shredder.sim.get_default_config()
        config['image']['bad_columns'] = True

        loader = shredx.sim.get_loader(tmpdir, config=config, rng=rng)
        shredx.shred_fofs(
            loader=loader,
            shredconf=conf,
            show=show,
            rng=rng,
            **kw
        )


if __name__ == '__main__':
    seed = 995
    test_shredding_smoke(seed, show=True)
    # test_shredding_bad_columns(seed, show=True)
