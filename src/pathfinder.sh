#!/bin/bash

relation=$1
python train_with_supervised_policy.py $relation
python train_with_reinforcement_policy.py $relation retrain
python train_with_reinforcement_policy.py $relation test

