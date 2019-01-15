import random
import numpy as np
import pandas

output_values1 = [ -1.0, 1.0 ]
output_values2 = [ -1.0, -0.33333333333333337, 0.33333333333333326, 1.0 ]
output_values4 = [-1.0, -0.8666666666666667, -0.7333333333333334, -0.6,
        -0.4666666666666667, -0.33333333333333337, -0.19999999999999996,
        -0.06666666666666665, 0.06666666666666665, 0.19999999999999996,
        0.33333333333333326, 0.46666666666666656, 0.6000000000000001,
        0.7333333333333334, 0.8666666666666667, 1.0]


def gen_lowcard_nom_dataset(n, nattr, seed, max_depth, card_range=[2, 16]):
    np.random.seed(seed)

    # generate columns
    columns = []
    for i in range(nattr):
        card = np.random.randint(card_range[0], card_range[1])
        columns.append(np.random.randint(0, card, n))

    # simulate a decision tree to generate output
    output = np.zeros(n)
    stack = []
    stack.append(np.array(range(n))) # node selection
    depths = [0]
    node_ids = [0]
    node_count = 0

    while stack:
        examples = stack.pop(-1)
        depth = depths.pop(-1)
        node_id = node_ids.pop(-1)
        node_count += 1

        print(" ITER: node_id {}, #ex {}, #stack {}, depth {}, #nodes {}".format(
            node_id, len(examples), len(stack), depth, node_count))

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

            print("SPLIT: node_id {}, column {}, split_val {}, #left {}, #right {}".format(
                node_id, column_j, split_val, len(left), len(right)))

            stack.append(right)
            stack.append(left) # left first
            depths.append(depth + 1) # right
            depths.append(depth + 1) # left
            node_ids.append(2 * node_id + 2)
            node_ids.append(2 * node_id + 1)
        else:
            #leaf_value = np.random.rand()
            leaf_value = random.choice(output_values1)
            print(" LEAF: node_id {} value {}".format(node_id, leaf_value))
            output[examples] = leaf_value

    # add some noise
    #output += 0.05 * np.random.randn(n)

    data = {}
    for (i, col) in enumerate(columns):
        data["col{}".format(i)] = col
    data["output"] = output.round(3)

    frame = pandas.DataFrame(data = data)
    return frame

if __name__ == "__main__":
    seed = 12
    n = 100000
    attr = 64
    max_depth = 4
    card_range = [4, 5]
    compression = False
    test_frac = 0.0

    #for attr in [4, 8, 16, 32, 64, 128, 256]:
    compr_opt = "gzip" if compression else None
    compr_ext = ".gz" if compression else ""

    ftrain = "/tmp/train{:03}-{}.csv{}".format(attr, n, compr_ext)
    ftest = "/tmp/test{:03}-{}.csv{}".format(attr, n, compr_ext)
    frame = gen_lowcard_nom_dataset(int(n * (1.0+test_frac)), attr, seed,
            max_depth=max_depth,
            card_range=card_range)

    print(ftrain)
    frame[0:n].to_csv(ftrain, header=True, index=False, compression=compr_opt)

    if test_frac > 0.0:
        print(ftest)
        frame[n:].to_csv(ftest, header=True, index=False, compression=compr_opt)
