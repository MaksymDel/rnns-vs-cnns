import csv
import json
""" Quick hacks here. Usage:
1) If we want to convert .txt to .jsonl:
python3 data_utils/predict_utils.py path_in.txt path_out.jsonl
2) If we want to convert .jsonl output to csv:
python3 data_utils/predict_utils.py path_in.jsonl path_out.csv 
"""
def txt2json_lines(path_in_file, path_out_file):
    with open(path_in_file, 'r') as f:
        txtlines = f.readlines()
    
    jsonlines = []
    for l in txtlines:
        jl = '{"sentence": "%s"}' % l[:-1]
        jsonlines.append(jl)

    with open(path_out_file, 'w', ) as f:
        f.write("\n".join(jsonlines))

def json_lines2csv(path_in_file, path_out_file):
    with open(path_in_file, 'r') as f:
        jsonlines = f.readlines()
        
    with open('data/test_public_idx.txt', 'r') as f:
        idxs = f.readlines()

    fw = csv.writer(open(path_out_file, "w"))
    fw.writerow(["id", "EAP", "HPL", "MWS"])
    
    for i in range(len(idxs)):
        x = json.loads(jsonlines[i])
        if i != len(idxs-1):
            id = idxs[i][:-1]
        else:
            id = idxs[i]
        fw.writerow([id, *tuple(x['class_probabilities'])])

import sys

args = sys.argv
if args[2][-6:] == '.jsonl':
    print('Generating %s from %s...' % (args[2], args[1]))
    txt2json_lines(args[1], args[2])
    print('Finished')
elif args[2][-4:] == '.csv':
    print('Generating %s from %s...' % (args[2], args[1]))
    json_lines2csv(args[1], args[2])
    print('Finished')
    
else:
    raise AttributeError("Usage should be: python3 data_utils/predict_utils.py path_in.txt path_out.jsonl") 