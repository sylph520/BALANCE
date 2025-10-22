import argparse


def main(input_file:str):
    with open(input_file, 'r') as f:
        sqls = f.readlines()
    # __import__('ipdb').set_trace()
    if sqls[-1] == '':
        sqls.pop()
    num_tpls = len(sqls)

    for i in range(1, num_tpls+1):
        input_fn = input_file.split('.')[0]
        qstr = sqls[i-1]
        output_fn = input_fn.upper() + f"_{i}.txt"
        with open(output_fn, 'w') as f:
            f.write(qstr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='tpcdsc.sql')
    args = parser.parse_args()
    main(args.input_file)
