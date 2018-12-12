import numpy as np
import pandas

def gen_lowcard_nom_dataset(n, seed):
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

if __name__ == "__main__":
    seed = 91
    n = 100
    filename = "/tmp/data{}.csv.gz".format(n)
    frame = gen_lowcard_nom_dataset(n, seed)

    print(filename)
    frame.to_csv(filename, header=True, index=False, compression="gzip")
