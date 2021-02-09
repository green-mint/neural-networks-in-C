#!/bin/sh

export DEMO_IMGS=$(ls ./demo_imgs | wc -l)
make predict_model
/home/green-mint/dev/.venv/bin/python3 /home/green-mint/dev/c-nn/create_data.py
clear
./predict_model
