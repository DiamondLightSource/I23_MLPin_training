import argparse

parser = argparse.ArgumentParser(description='Remove lines containing ">" from a file.')
parser.add_argument('input_file', help='The path to the file to strip')

args = parser.parse_args()

with open(args.input_file, 'r') as infile:
    lines = infile.readlines()

with open(args.input_file, 'w') as outfile:
    for line in lines:
        if '>' not in line:
            outfile.write(line)