#!/bin/bash
cd models
python BM25.py
python RankNet.py
python tf_ranking.py
# display TF-Ranking log results on Tensorboard
tensorboard --logdir tf_ranking_logs
# show TF-Ranking log results
xdg-open http://localhost:6006/
# back to root dir
cd ..