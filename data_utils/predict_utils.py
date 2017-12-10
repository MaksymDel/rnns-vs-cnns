""" Quick hacks here. Usage:
1) If we want to convert .txt to .jsonl:
python3 data_utils/predict_utils.py path_in.txt path_out.jsonl
2) If we want to convert .jsonl output to txt:
python3 data_utils/predict_utils.py path_in.jsonl path_out.txt 
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

def json_lines2txt(path_in_file, path_out_file):
    with open(path_in_file, 'r') as f:
        jsonlines = f.readlines()
    
    txtlines = []
    for l in jsonlines:
        tl = l[-6:-3]
        txtlines.append(tl)

    with open(path_out_file, 'w', ) as f:
        f.write("\n".join(txtlines))

import sys

args = sys.argv
if args[2][-6:] == '.jsonl':
    print('Generating %s from %s...' % (args[2], args[1]))
    txt2json_lines(args[1], args[2])
    print('Finished')
elif args[2][-4:] == '.txt':
    print('Generating %s from %s...' % (args[2], args[1]))
    json_lines2txt(args[1], args[2])
    print('Finished')
    
else:
    raise AttributeError("Usage should be: python3 data_utils/predict_utils.py path_in.txt path_out.jsonl") 