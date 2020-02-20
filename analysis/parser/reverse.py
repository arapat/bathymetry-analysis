
s = ''
while True:
    try:
        line = input()
    except:
        break
    s += line + '\n'

k = 0
for line in s.split('\n'):
    elems = line.split()
    for t in elems:
        try:
            if t[0] == '{' and t[-1] == '}':
                t = "{ %.2f }" % (100.0 - float(t[1:-1]))
            elif t[0] == '{':
                t = "{ %.2f" % (100.0 - float(t[1:]))
            elif t[-1] == '}':
                t = "%.2f }" % (100.0 - float(t[:-1]))
            else:
                t = "%.2f" % (100.0 - float(t))
            k += 1
        except:
            pass
        print(t + " ", end='')
    print("")

print(k)
