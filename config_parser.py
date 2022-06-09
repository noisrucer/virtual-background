import os
from pathlib import Path
import logging

from utils import read_json, write_json
from logger import setup_logging


class ConfigParser:
    def __init__(self, args):
        # [1] resume config.json or new config.json
        if args.resume is not None: # resume config.json
            resume = Path(args.resume)
            cfg_path = resume
        else: # new config.json
            assert args.config is not None, 'Provide at least --config or --resume'
            resume = None
            cfg_path = Path(args.config)

        if args.config and resume: # Cannot have both
            raise Exception("Provide only one of --config and --resume")

        # [2] Load config
        config = read_json(cfg_path)

        self._config = self._update_config(config, args)
        self.resume = resume

        # [3] settings for saving records
        save_dir = Path(self.config['trainer']['save_dir'])
        run_id = self.config['run_id']

        self._model_dir = save_dir / 'models' / run_id
        self._log_dir = save_dir / 'log' / run_id

        # [4] Create model/log dirs to save records
        self.model_dir.mkdir(parents=True, exist_ok=False)
        self.log_dir.mkdir(parents=True, exist_ok=False)

        write_json(self.config, self.model_dir / 'config.json')

        # [5] Logging
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }


    def init_obj(self, module, cfg_type, *args, **kwargs):
        module_name = self.config[cfg_type]['type']
        module_args = self.config[cfg_type]['args']

        assert all([k not in module_args for k in kwargs]), 'kwargs cannot override config args'
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)


    def get_logger(self, name, verbosity=2):
        assert verbosity in self.log_levels, "Invalid verbosity option: {}. Valid options are {}".format(verbosity, self.log_levels.keys())
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger


    def _update_config(self, config, args):
        if args.loss is not None:
            config['loss'] = args.loss

        if args.epochs is not None:
            config['trainer']['epochs'] = args.epochs

        if args.learning_rate is not None:
            config['optimizer']['args']['lr'] = args.learning_rate

        if args.batch_size is not None:
            config['train_loader']['args']['batch_size'] = args.batch_size
            config['val_loader']['args']['batch_size'] = args.batch_size

        return config

    def __getitem__(self, name):
        return self.config[name]

    @property
    def config(self):
        return self._config

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def log_dir(self):
        return self._log_dir
