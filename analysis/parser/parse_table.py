import sys


with open(sys.argv[1]) as f:
    for line in f:
        for eg in line.split():
            try:
                t = float(eg)
                print("%.2f &" % (t * 100.0), end=' ')
            except:
                print(eg.strip() + " &", end=' ')
        print('\b\b\\\\')

