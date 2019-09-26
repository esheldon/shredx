from pprint import pformat
import logging
from copy import deepcopy
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'fofs': {
        'min_fofsize': 2,
    },
    'data': {
        'zero_weight_badpix': 0,
    },
    'guess': {
        'model': 'dev',
    },
    'shredding': {
        'psf_ngauss': 5,
    },
}


def get_config(*, config_file=None):
    """
    If a config file is sent, read the configuration.  Add in
    defaults as needed.  Otherwise use the default config.
    """
    if config_file is None:
        conf = deepcopy(DEFAULT_CONFIG)
    else:
        logger.info('reading config: %s' % config_file)
        with open(config_file) as fobj:
            conf = yaml.load(fobj, Loader=yaml.Loader)

        if 'fofs' not in conf:
            conf['fofs'] = deepcopy(DEFAULT_CONFIG['fofs'])

        if 'shredding' not in conf:
            conf['shredding'] = deepcopy(DEFAULT_CONFIG['shredding'])

        if 'guess' not in conf:
            conf['guess'] = deepcopy(DEFAULT_CONFIG['guess'])

        if 'data' not in conf:
            conf['data'] = deepcopy(DEFAULT_CONFIG['data'])

        assert 'min_fofsize' in conf['fofs']
        assert 'psf_ngauss' in conf['shredding']
        assert 'model' in conf['guess']
        assert 'zero_weight_badpix' in conf['data']

        logger.info('final config: \n%s' % pformat(conf))

    return conf
