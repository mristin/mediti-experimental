#!/usr/bin/env python3

"""Delete the training instance."""

import inspect
import os
import sys
import pathlib
import re
import argparse
from typing import List, Union, Set, Dict
import datetime
import subprocess

INSTANCE_NAME = "mediti-train"

def main() -> None:
    """Execute the main routine."""
    print("Deleting the instance...")

    subprocess.check_call([
        'gcloud', 'compute', 'instances', 'delete', INSTANCE_NAME])

if __name__ == "__main__":
    main()
