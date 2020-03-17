import sys

num_rows = 7
num_cols = 8

with open(sys.argv[1]) as f:
    for row in range(num_rows):
        #if row == num_rows - 1:
        #    print("\\hline")
        for col in range(num_cols):
            line = f.readline()
            vals = line.split()
            row_name = vals[0].strip()
            value = float(vals[-1])
            if col == 0:
                print("{} ".format(row_name.replace("_", " ")), end='')
            if col == row:
                print("& {\color{blue} %.2f } " % (value * 100.0), end='')
            else:
                print("& %.2f " % (value * 100.0), end='')
        print("\\\\")
print("\\hline")
