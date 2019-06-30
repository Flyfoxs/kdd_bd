#!/usr/bin/env bash
cd "$(dirname "$0")"
cd ..

mkdir -p ~/MindRank
mkdir -p ~/MindRank/data
mkdir -p ~/MindRank/model
mkdir -p ~/MindRank/result
mkdir -p ~/MindRank/output/stacking
mkdir -p ~/MindRank/output/sub


mkdir -p ./
mkdir -p ./data
mkdir -p ./model
mkdir -p ./result
mkdir -p ./output/stacking
mkdir -p ./output/sub
mv ./input/*  ./data
mv ./data/data_set_phase2/* ./data




