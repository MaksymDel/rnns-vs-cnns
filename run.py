#!/usr/bin/env python3
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

# NOTE: This line is important!  It's what makes all of the classes in `my_library` findable by
# AllenNLP's registry.
from rnns_vs_cnns import *

from allennlp.commands import main  # pylint: disable=wrong-import-position

predictors = {
    'author_classifier': 'author_classifier'
}

if __name__ == "__main__":
    main(prog="python3 run.py", predictor_overrides=predictors)