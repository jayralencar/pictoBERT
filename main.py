import sys
import os

if len(sys.argv) > 1:
    _, action = sys.argv
    if action == "test":
        if len(sys.argv) == 2:
            form = "pretrained"
        else:
            form = sys.argv[2]
        if form == "pretrained":
            