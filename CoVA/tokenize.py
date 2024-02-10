from tokenizer import *

import sys

tokenizer = Tokenizer()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Argument is missing. (e.g. python3 tokenize.py <input directory> <output directory>)")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    tokenizer(input_dir, output_dir)