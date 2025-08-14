import argparse
import re
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str)

args = parser.parse_args()

MAX_LENGTH = 150
MIN_LENGTH = 3
LENGTH_RATIO = 7

lines_1_clean = []
lines_2_clean = []

with open(args.path1) as f1:
    lines_1 = f1.readlines()
    with open(args.path2) as f2:
        lines_2 = f2.readlines()
        # store indexes of any duplicates 
        print("Length of files before processing: ", len(lines_1))
        assert len(lines_1) == len(lines_2), f"{args.path1} and {args.path2} do not have same number of lines"
        
        idxs_to_remove = []
        tuple_set = set()
        for idx in range(len(lines_1)):
            src_and_target = (lines_1[idx], lines_2[idx])
            if src_and_target in tuple_set:
                idxs_to_remove.append(idx)
            else:
                tuple_set.add(src_and_target)
        print("Number of duplicates in test set: ", len(idxs_to_remove))
                
        for i, (x, y) in enumerate(zip(lines_1, lines_2)):
            # remove duplicates
            if i in idxs_to_remove:
                continue
            # split "OR" english sentences 
            if " OR: " in x or "(OR: " in x:
                x1, x2 = x.split("OR: ")[0:2]
                # pick first option
                lines_1_clean.append(x1 + '\n')
                lines_2_clean.append(y)
            else:
                lines_1_clean.append(x)
                lines_2_clean.append(y)
    
assert len(lines_1_clean) == len(lines_2_clean), f"{args.path1} and {args.path2} cleaned do not have same number of lines"
print("Length of files after processing: ", len(lines_1_clean))

with open(args.path1+'_filtered', "w") as outf1:
    with open(args.path2+'_filtered', "w") as outf2:
        for (line_1, line_2) in zip(lines_1_clean, lines_2_clean):
            outf1.write(line_1)
            outf2.write(line_2)
