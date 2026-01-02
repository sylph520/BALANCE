import argparse

def main(args):
    input_fn = args.input_fn
    with open(input_fn, 'r') as f:
        sqls = f.readlines()
    round_size = args.round_size
    assert len(sqls) % round_size == 0, f"the number of sqls {len(sqls)} is not the times of the round size {round_size}"
    num_rounds = int(len(sqls) / round_size)
    # __import__('ipdb').set_trace()
    output_fnbase = args.output_fnbase
    for i in range(num_rounds):
        wk = sqls[i*round_size:(i+1)*round_size]
        output_fn = f"{output_fnbase}{i+1}.txt"
        with open(output_fn, 'w') as f:
            f.write(''.join(wk))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', type=str, default='workload_combine.txt')
    parser.add_argument('--output_fnbase', type=str, default='workload')
    parser.add_argument('--round_size', type=int, default=16)
    args = parser.parse_args()
    main(args)
