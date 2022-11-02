#!/bin/bash

virtualenv -q -p /usr/bin/python3.9 $1
source $1/bin/activate

$1/bin/pip install aiogram
python3 ../telegram_bot/run.py
