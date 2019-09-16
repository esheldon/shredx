import logging
import numpy as np
import pytest
import shredx
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('seed', [2510, 21064])
def test_shredding_smoke(seed, show=False, **kw):
    """
    test we can run the fof finder and extract the group
    """

    shredx.setup_logging('debug')
    rng = np.random.RandomState(seed)

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        shredx.shred_fofs(loader=loader, show=show, **kw)


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

    with TemporaryDirectory() as tmpdir:

        loader = shredx.sim.get_loader(tmpdir, rng=rng)
        shredders = shredx.shred_fofs(
            loader=loader,
            show=show,
            get_shredders=True,
            **kw
        )

        for s in shredders:

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
