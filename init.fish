#!/bin/fish
set SCRIPT_DIR (realpath (dirname (status --current-filename)))
echo $SCRIPT_DIR

# Change this to your desired directory:
set --export WS_DIR $SCRIPT_DIR/workspace

# Stuff that ALFRED needs
set --export ALFRED_PARENT_DIR $WS_DIR/alfred_src
set --export ALFRED_ROOT $ALFRED_PARENT_DIR/alfred

set --export PYTHONPATH $PYTHONPATH:$ALFRED_PARENT_DIR:$ALFRED_ROOT:$ALFRED_ROOT/gen

# Stuff that HLSM needs
set --export LGP_WS_DIR "$WS_DIR"
set --export LGP_MODEL_DIR "$WS_DIR/models"
set --export LGP_DATA_DIR "$WS_DIR/data"
