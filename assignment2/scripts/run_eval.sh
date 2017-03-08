#!/bin/bash
python3 similarity.py > prediction.csv --embedding ../embeddings.txt --words similarity/dev_x.csv
python3 evaluate.py --predicted prediction.csv --development similarity/dev_y.csv

