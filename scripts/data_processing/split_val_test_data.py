#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 10:24:00 2025

@author: s2583833

"""

import argparse
import re
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str)

args = parser.parse_args()

with open(args.path1) as f1:
    lines_1 = f1.readlines()
    with open(args.path2) as f2:
        lines_2 = f2.readlines()
        

combined = list(zip(lines_1, lines_2))
random.shuffle(combined)
lines_1_shuffled, lines_2_shuffled = zip(*combined)

# split out either 10% or up to 1000

# split out 10% 
num_lines = len(lines_1_shuffled)
ten_percent = int(num_lines/10)
twenty_percent = ten_percent*2
if ten_percent > 1000:
    val_lines_1 = lines_1_shuffled[0:1000]
    test_lines_1 = lines_1_shuffled[1000:2000]
    train_lines_1 = lines_1_shuffled[2000:]
    val_lines_2 = lines_2_shuffled[0:1000]
    test_lines_2 = lines_2_shuffled[1000:2000]
    train_lines_2 = lines_2_shuffled[2000:]
else:
    val_lines_1 = lines_1_shuffled[0:ten_percent]
    test_lines_1 = lines_1_shuffled[ten_percent:twenty_percent]
    train_lines_1 = lines_1_shuffled[twenty_percent:]
    val_lines_2 = lines_2_shuffled[0:ten_percent]
    test_lines_2 = lines_2_shuffled[ten_percent:twenty_percent]
    train_lines_2 = lines_2_shuffled[twenty_percent:]


with open(args.path1+'_1000val', "w") as outf1:
    with open(args.path2+'_1000val', "w") as outf2:
        for (line_1, line_2) in zip(val_lines_1, val_lines_2):
            outf1.write(line_1)
            outf2.write(line_2)
            
with open(args.path1+'_1000test', "w") as outf1:
    with open(args.path2+'_1000test', "w") as outf2:
        for (line_1, line_2) in zip(test_lines_1, test_lines_2):
            outf1.write(line_1)
            outf2.write(line_2)
            
with open(args.path1, "w") as outf1:
    with open(args.path2, "w") as outf2:
        for (line_1, line_2) in zip(train_lines_1, train_lines_2):
            outf1.write(line_1)
            outf2.write(line_2)