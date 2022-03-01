#!/bin/bash

remove_dir() {
    for DIRNAME in $(find . -type d -name $1); do
        rm -dr $DIRNAME
        echo "remove directory: $DIRNAME"
    done
}

remove_dir ".ipynb_checkpoints"
remove_dir "__pycache__"
remove_dir ".mypy_cache"
remove_dir ".vscode"
