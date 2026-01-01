input_fn = 'ceb16.sql'

with open(input_fn, 'r') as f:
    sqls = f.readlines()
if sqls[-1] == '':
    sqls.pop()
num_sqls = len(sqls)
print(f"load {num_sqls} sqls")

for i in range(num_sqls):
    with open(f"CEB_{i+1}.txt", 'w') as f:
        f.write(sqls[i])

