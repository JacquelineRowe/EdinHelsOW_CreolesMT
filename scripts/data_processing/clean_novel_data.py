import argparse
import re
import random

parser = argparse.ArgumentParser()

# where paths 1 and 2 are the files to be cleaned, and path 3 and 4 are the kreyolmt data to compare with 
parser.add_argument('--path1', type=str)
parser.add_argument('--path2', type=str)
parser.add_argument('--path3', type=str)
parser.add_argument('--path4', type=str)

args = parser.parse_args()

src_train_lines = []
tgt_train_lines = []

def clean_line(line):
    line = re.sub(r' \[\.{3,}\]', '', line)
    line = re.sub(r' \[…\]', '', line)
    line = re.sub(r'\[\.{3,}\] ', '', line)
    line = re.sub(r'\[…\] ', '', line)
    line = re.sub(r'[^\w\s]', '', line)

    return line.lower().strip()

with open(args.path3) as f1:
    src_train_lines_all = f1.readlines()
    for line in src_train_lines_all:
        line_clean = clean_line(line)
        src_train_lines.append(line_clean)
with open(args.path4) as f1:
    tgt_train_lines_all = f1.readlines()
    for line in tgt_train_lines_all:
        line_clean = clean_line(line)
        tgt_train_lines.append(line_clean)

# print(src_train_lines)
# print(tgt_train_lines)
# print("Length of kreyolmt source file", len(src_train_lines))
# print("Length of kreyolmt target file", len(tgt_train_lines))

# create tuple set of kreyol-mt data 
src_set = set(item.strip().lower() for item in src_train_lines)
tgt_set = set(item.strip().lower() for item in tgt_train_lines)

with open(args.path1) as f3:
    new_lines_1 = f3.readlines()
with open(args.path2) as f4:
    new_lines_2 = f4.readlines()

# check lines_1 for overlap with src, and lines_2 for overlap with tgt 
assert len(new_lines_1) == len(new_lines_2), f"{args.path1} and {args.path2} do not have same number of lines"
idxs_to_remove = []

for idx in range(len(new_lines_1)):
    line1_clean = clean_line(new_lines_1[idx])
    line2_clean = clean_line(new_lines_2[idx])
    if line1_clean.strip() in src_set or line2_clean.strip() in tgt_set: 
        idxs_to_remove.append(idx)
        print(line1_clean.strip())
        print(line2_clean.strip())
    else:
        pass
print(len(idxs_to_remove))


lines_1_clean = []
lines_2_clean = []
    
for i, (x, y) in enumerate(zip(new_lines_1, new_lines_2)):
    # remove duplicates
    if i in idxs_to_remove:
        continue
    else:
        lines_1_clean.append(x)
        lines_2_clean.append(y)
    
assert len(lines_1_clean) == len(lines_2_clean), f"{args.path1} and {args.path2} cleaned do not have same number of lines"
# print("Length of our set after processing: ", len(lines_1_clean))

with open(args.path1, "w") as outf1:
    with open(args.path2, "w") as outf2:
        for (line_1, line_2) in zip(lines_1_clean, lines_2_clean):
            outf1.write(line_1)
            outf2.write(line_2)
