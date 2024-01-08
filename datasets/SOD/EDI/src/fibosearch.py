def fibosearch(fhandle, a, b, npoints):
    nfibo = 22
    fibo = [1, 1] + [0] * (nfibo - 2)
    for k in range(1, nfibo - 1):
        fibo[k + 1] = fibo[k] + fibo[k - 1]

    fiboindex = 3
    while fibo[fiboindex] < npoints:
        fiboindex += 1

    for k in range(fiboindex - 1):
        if k == 0:
            x1 = a + fibo[fiboindex - k - 1] / fibo[fiboindex - k + 1] * (b - a)
            x2 = b - fibo[fiboindex - k - 1] / fibo[fiboindex - k + 1] * (b - a)
            fx1 = fhandle(x1)
            fx2 = fhandle(x2)

        if fx1 < fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + fibo[fiboindex - k - 1] / fibo[fiboindex - k + 1] * (b - a)
            fx1 = fhandle(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = b - fibo[fiboindex - k - 1] / fibo[fiboindex - k + 1] * (b - a)
            fx2 = fhandle(x2)

    if fx1 < fx2:
        x = x1
    else:
        x = x2

    return x



