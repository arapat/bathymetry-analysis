import sys


with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        for j, eg in enumerate(line.split(',')):
            try:
                t = float(eg)
                val = "%.2f" % (t * 100.0)
                if i + 1 == j:
                    print("{ \\color{blue} \\textbf{" + val + "} } &", end=' ')
                else:
                    print(val + " &", end=' ')
            except:
                print(eg.strip() + " &", end=' ')
        print('\b\b\\\\')

