import argparse
import os
import glob
import random

from main import main

num_samples = 100
print 'we have %s samples' % num_samples

parser = argparse.ArgumentParser(
    description="compare two sets")
parser.add_argument('first',type=str, help="dir to first set")
parser.add_argument('firstname',type=str, help="name of first set")
parser.add_argument('second',type=str, help="dir to second set")
parser.add_argument('secondname',type=str, help="name of second set")
parser.add_argument('comparison',type=str, help="comparison folder/name")
args = parser.parse_args()

globstr1 = os.path.join(args.first, 'temp*.mid')
set1 = glob.glob(
    globstr1)

if len(set1) > num_samples:
    random.shuffle(set1)
    set1 = set1[:num_samples]

set1name = args.firstname
num_samples = len(set1)

globstr2 = os.path.join(args.second, 'temp*.mid')
set2 = glob.glob(globstr2)

if len(set2) > num_samples:
    random.shuffle(set2)
    set2 = set2[:num_samples]

set2name = args.secondname

dstfolder = args.comparison
if not os.path.exists(dstfolder):
    os.mkdir(dstfolder)

with open(os.path.join(dstfolder, '0info.txt'), 'w') as f:
    f.writelines("Comparison between %s and %s" %(globstr1, globstr2))

main(set1, set2, set1name, set2name, dstfolder) # order matters in KL
