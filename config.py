import datetime
import logging
import random
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from functools import partial
from io import StringIO
from pathlib import Path
from typing import ClassVar, Optional

today_str = lambda: (datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

root_logger = logging.getLogger()
from utils import requires_import 

@dataclass
class Config:
    """ A base-class that implements most of the usual configuration options.
    
    Configuration options differ from HyperParameters, as they are not being
    optimized over, and are not mutated by the EPBT algorithm.
    
    NOTE: we create these options in a dataclass in order to use
    [`simple-parsing`](https://github.com/lebrice/SimpleParsing)
    to add all the arguments directly.
    """
    # Run in debug mode.
    debug: bool = False
    # random seed
    seed: Optional[int] = None
    # data directory
    data_dir: Path = Path("data")
    # Directory containing the logs of each run.
    root_log_dir: Path = Path("logs")

    # Name to give to this particular run.
    run_name: str = field(default_factory=today_str)
    # how many batches to wait before logging training status
    log_interval: int = 10
    
    # disables CUDA training
    no_cuda: bool = False
    
    _root_logger_setup: ClassVar[bool] = False

    ## Non-parsed fields (created in __post_init__):
    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            with requires_import("torch") as torch:
                torch.manual_seed(self.seed)
            with requires_import("numpy") as np:
                np.random.seed(self.seed)

        with requires_import("torch") as torch:
            cuda_available = torch.cuda.is_available()
            use_cuda = not self.no_cuda and cuda_available

            self.device = torch.device("cuda" if use_cuda else "cpu")   
            self.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        if self.debug:
            self.log_dir = self.root_log_dir / "debug"
        else:
            self.log_dir = get_new_file(self.root_log_dir / self.run_name)
        self.log_file = (self.log_dir / "log.txt")
        self.err_file = (self.log_dir / "error.txt")

    def setup_root_logger(self):
        """ Setup the root logger. 
        NOTE: Trying to follow https://docs.python.org/3/howto/logging-cookbook.html
        """
        if Config._root_logger_setup:
            return

        level = logging.DEBUG if self.debug else logging.INFO
        # root_logger.setLevel(level)
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        debug_formatter = logging.Formatter(
            "%(levelname)s: "
            "%(name)s - "
            "%(asctime)s - "
            "./%(filename)s:%(lineno)d:\t"
            "%(funcName)s(): "
            "%(message)s"
        )
        fh.setFormatter(debug_formatter)
        root_logger.addHandler(fh)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.debug else logging.INFO)
        console_formatter = logging.Formatter(
            "%(levelname)s: "
            "%(name)s - "
            "./%(filename)s:%(lineno)d:\t"
            "%(funcName)s(): "
            "%(message)s"
            )
        ch.setFormatter(console_formatter)
        root_logger.addHandler(ch)

        # Add handler for TQDM progress bars.
        root_logger.addHandler(TqdmLoggingHandler())
        Config._root_logger_setup = True
        
    def get_logger(self, name: str=None) -> logging.Logger:
        self.setup_root_logger()
        if name is None:
            return root_logger
        else:
            return root_logger.getChild(name)

    @contextmanager
    def redirect_to_log_files(self):
        """Redirects stdout and stderr to `self.log_file` and `self.err_file`  files, respectively.
        """
        self.logger.debug(f"Redirecting stdout to {self.log_file} and stderr to {self.err_file}.")
        err_buf = StringIO()

        with open(self.log_file, "a") as out:
            with redirect_stdout(out), redirect_stderr(err_buf):
                yield

        with err_buf:
            # if the stderr buffer isn't empty, dump the contents to the file.
            if err_buf.tell() != 0:
                err_buf.seek(0)
                with open(self.err_file, "a") as err_file:
                    err_file.writelines(err_buf.readlines())


def get_new_file(file: Path) -> Path:
    """Creates a new file, adding _{i} suffixes until the file doesn't exist.
    
    Args:
        file (Path): A path.
    
    Returns:
        Path: a path that is new. Might have a new _{i} suffix.
    """
    if not file.exists():
        return file
    else:
        i = 0
        file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        while file_i.exists():
            i += 1
            file_i = file.with_name(file.stem + f"_{i}" + file.suffix)
        file = file_i
    return file

try:
    import tqdm
        
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)  
except ImportError:
    pass
