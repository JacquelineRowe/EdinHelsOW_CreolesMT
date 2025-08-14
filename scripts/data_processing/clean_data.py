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

or_sentences_count = 0
or_sentences_not_added_count = 0

with open(args.path1) as f1:
    lines_1 = f1.readlines()
    with open(args.path2) as f2:
        lines_2 = f2.readlines()
        print("Length of files before processing: ", len(lines_1))
        assert len(lines_1) == len(lines_2), f"{args.path1} and {args.path2} do not have same number of lines"
        # store indexes of any duplicates 
        idxs_to_remove = []
        tuple_set = set()
        for idx in range(len(lines_1)):
            src_and_target = (lines_1[idx], lines_2[idx])
            if src_and_target in tuple_set:
                idxs_to_remove.append(idx)
            else:
                tuple_set.add(src_and_target)
        # find acceptable tolerance for difference between length in soruce and target sentences - adapted from kreyolMT code 
        squares1, sum1, squares2, sum2 = 0, 0, 0, 0
        for idx in range(len(lines_1)):       
            sub1 = len(lines_1[idx]) / len(lines_2[idx])
            sum1 += sub1
            squares1 += sub1*sub1

            sub2 = len(lines_1[idx]) / len(lines_2[idx])
            sum2 += sub2
            squares2 += sub2*sub2

        n = len(lines_1)
        mean1 = sum1/n
        var1 = squares1/n - mean1**2
        sd1 = var1**0.5
        tol1 = LENGTH_RATIO*sd1
        #print("Tol 1", tol1)

        mean2 = sum2/n
        var2 = squares2/n - mean2**2
        sd2 = var2**0.5
        tol2 = LENGTH_RATIO*sd2
        #print("Tol 2", tol2)
        
        for i, (x, y) in enumerate(zip(lines_1, lines_2)):
            # remove duplicates
            if i in idxs_to_remove:
                continue
            # filter too long
            elif len(x.strip().split()) > MAX_LENGTH or len(y.strip().split()) > MAX_LENGTH:
                continue
            # remove source = target
            elif x == y:
                continue 
            # remove large difference between source and target sentences lengths 
            if abs(len(x)/len(y) - mean1) > tol1 or abs(len(y)/len(x) - mean2) > tol2:
                continue
            
            # delete sentence initial full stops 
            x = re.sub(r'^\.', '', x)
            y = re.sub(r'^\.', '', y)

            # replace any hex characters (mainly for bible data)
            x = re.sub(r'&#x27;', '\'', x)
            y = re.sub(r'&#x27;', '\'', y)
            
            # split "OR" english sentences 
            if x.count(" OR: ") == 1:
                or_sentences_count += 1
                x1, x2 = x.split(" OR: ")[0:2]
                # remove too short 
                if len(x1.strip().split()) < MIN_LENGTH or len(y.strip().split()) < MIN_LENGTH:
                    or_sentences_not_added_count += 1
                    continue
                else:
                    lines_1_clean.append(f"{x1} \n")
                    lines_2_clean.append(y)
                    
                if len(x2.strip().split()) < MIN_LENGTH or len(y.strip().split()) < MIN_LENGTH:
                    or_sentences_not_added_count += 1
                    continue
                else:
                    lines_1_clean.append(x2)
                    lines_2_clean.append(y)
                    or_sentences_count += 1
            elif x.count(" OR: ") > 1:
                continue
            else:
                    # remove too short 
                if len(x.strip().split()) < MIN_LENGTH or len(y.strip().split()) < MIN_LENGTH:
                    continue
                else:
                    lines_1_clean.append(x)
                    lines_2_clean.append(y)
    
added_or_sentences = or_sentences_count-or_sentences_not_added_count
assert len(lines_1_clean) == len(lines_2_clean), f"{args.path1} and {args.path2} cleaned do not have same number of lines"
print("Length of files after processing: ", len(lines_1_clean))
print("Number of additional sentences extracted from OR: sentences: ", added_or_sentences)
print("Lines removed during cleaning: ", len(lines_1) + added_or_sentences - len(lines_1_clean))

combined = list(zip(lines_1_clean, lines_2_clean))
random.shuffle(combined)
lines_1_clean, lines_2_clean = zip(*combined)

with open(args.path1+'_filtered', "w") as outf1:
    with open(args.path2+'_filtered', "w") as outf2:
        for (line_1, line_2) in zip(lines_1_clean, lines_2_clean):
            outf1.write(line_1)
            outf2.write(line_2)
