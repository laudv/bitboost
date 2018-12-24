import numpy as np
import pandas

def gen_lowcard_nom_dataset1(n, seed):
    np.random.seed(seed)
    data = {
        "col1": np.random.randint(0, 16, n),
        "col2": np.random.randint(0, 8, n),
        "col3": np.random.randint(0, 4, n),
        "col4": np.random.randint(0, 3, n),
    }

    output1 = 5.8*data["col1"] - 2.25*data["col2"] + 0.25*(data["col2"]-data["col3"])
    output2 = 2.8*data["col1"] + 3.25*data["col2"] + 1.25*(data["col2"]-data["col3"])
    output3 = 9.8*data["col1"] - 1.25*data["col2"] - 5.25*(data["col2"]-data["col3"])
    outputs = [output1, output2, output3]

    output = np.array(list(map(lambda p: outputs[p[1]][p[0]], enumerate(data["col4"]))))
    output += 0.01 * np.random.randn(n)

    data["output"] = output

    frame = pandas.DataFrame(data=data)
    return frame

def gen_lowcard_nom_dataset2(n, nattr, seed):
    np.random.seed(seed)

    # generate columns
    columns = []
    for i in range(0, nattr):
        card = np.random.randint(2, 16)
        columns.append(np.random.randint(0, card, n))

    output = np.zeros(n)
    for i in np.random.permutation(range(0, n)):
        if output[i] != 0.0:
            if np.random.rand() > 0.05: continue
        col = np.random.randint(0, nattr)
        val = columns[col][i]
        outval = np.random.rand();
        output[columns[col] == val] = outval
    output += 0.01 * np.random.randn(n)
    output = output.round(3)

    data = {}
    for (i, col) in enumerate(columns):
        data["col{}".format(i)] = col
    data["output"] = output

    frame = pandas.DataFrame(data = data)
    return frame

def gen_lowcard_nom_dataset3(n, nattr, seed, max_depth):
    np.random.seed(seed)

    # generate columns
    columns = []
    for i in range(nattr):
        card = np.random.randint(2, 16)
        columns.append(np.random.randint(0, card, n))

    # simulate a decision tree to generate output
    output = np.zeros(n)
    stack = []
    stack.append(np.array(range(n))) # node selection
    depths = [0]
    node_count = 0

    while stack:
        examples = stack.pop(-1)
        depth = depths.pop(-1)
        node_count += 1

        print(" ITER: #ex {}, #stack {}, depth {}, #nodes {}".format(len(examples),
            len(stack), depth, node_count))

        if depth < max_depth:
            if len(examples) == 0: continue

            max_tries = 16
            for i in range(max_tries): # try 16 times for a non-zero split
                column_j = np.random.randint(0, nattr)
                values = columns[column_j][examples]

                split_val = values[np.random.randint(0, len(values))]

                left = examples[values == split_val]
                if len(left) > 0 and len(left) < len(examples): break
                print(" NON-ZERO SPLIT FAIL {} col{}={}".format(i, column_j, split_val),
                        "zero it is" if i==max_tries-1 else "")

            right = examples[values != split_val]

            print("SPLIT: column {}, split_val {}, #left {}, #right {}".format(
                column_j, split_val, len(left), len(right)))

            stack.append(right)
            stack.append(left) # left first
            depths.append(depth + 1) # right
            depths.append(depth + 1) # left
        else:
            leaf_value = np.random.rand()
            print(" LEAF: value {}".format(leaf_value))
            output[examples] = leaf_value

    output += 0.05 * np.random.randn(n)

    data = {}
    for (i, col) in enumerate(columns):
        data["col{}".format(i)] = col
    data["output"] = output.round(3)

    frame = pandas.DataFrame(data = data)
    return frame

def gen_simple_lowcard_nom_dataset(n, seed):
    np.random.seed(seed)
    data = {
        "col1": np.random.randint(0, 4, n),
        "col2": np.random.randint(0, 2, n),
    }

    output1 = 0.25*data["col1"]
    output2 = 0.50*data["col2"]
    outputs = [output1, output2]

    output = np.array(list(map(lambda p: outputs[p[1]][p[0]], enumerate(data["col2"]))))
    output += 0.005 * np.random.randn(n)

    data["output"] = output

    frame = pandas.DataFrame(data=data)
    return frame

if __name__ == "__main__":
    seed = 91
    n = 100000
    attr = 4
    compression = False
    test_frac = 0.0

    #for attr in [4, 8, 16, 32, 64, 128, 256]:
    compr_opt = "gzip" if compression else None
    compr_ext = ".gz" if compression else ""

    ftrain = "/tmp/train{:03}-{}.csv{}".format(attr, n, compr_ext)
    ftest = "/tmp/test{:03}-{}.csv{}".format(attr, n, compr_ext)
    frame = gen_lowcard_nom_dataset3(int(n * (1.0+test_frac)), attr, seed, max_depth=5)

    print(ftrain)
    frame[0:n].to_csv(ftrain, header=True, index=False, compression=compr_opt)

    if test_frac > 0.0:
        print(ftest)
        frame[n:].to_csv(ftest, header=True, index=False, compression=compr_opt)
