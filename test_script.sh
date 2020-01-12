#!/bin/bash

case "$1" in
  "TF1")
    pip intstall -r requirements_tf1.txt
    python -m pytest
  ;;
  "TF2")
    pip intstall -r requirements_tf2.txt
    python -m pytest
  ;;
esac