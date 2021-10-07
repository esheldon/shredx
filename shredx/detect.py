"""
some basic detection using sep
"""

import numpy as np


def run_sep(image, noise):
    """
    run sep (sextractor) on the image and get the catalog and seg map
    """
    import sxdes

    cat, seg = sxdes.run_sep(image, noise)
    return cat, seg


def run_sep_on_mbobs(mbobs):
    """
    run sep and get a cat and seg map
    """

    import shredder

    coadd_obs = shredder.coadding.make_coadd_obs(mbobs)

    noise = np.sqrt(1.0/coadd_obs.weight[0, 0])
    with coadd_obs.writeable():
        result = run_sep(coadd_obs.image, noise)

    return result
