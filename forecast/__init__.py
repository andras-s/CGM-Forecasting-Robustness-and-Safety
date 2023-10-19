import os
from .config import args
if not os.path.exists(args.base_dir):
    os.makedirs(args.base_dir)

from forecast import config, data, helper, loss, model, train