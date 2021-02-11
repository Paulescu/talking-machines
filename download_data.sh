#!/bin/bash
dir=$1
wget https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json -P ${dir}