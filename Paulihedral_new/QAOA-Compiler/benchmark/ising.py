import numpy as np
from qiskit.optimization.applications.ising import max_cut
from .mypauli import pauliString
w1 = np.array([[0., 1., 1., 1.], [1., 0., 1., 0.], [1., 1., 0., 1.], [1., 0., 1., 0.]])

def gene_dot_1d(w, interaction='Z'):
    nq = w + 1
    oplist = []
    for i in range(nq - 1):
        ps = i*'I' + interaction + interaction + (nq-2-i)*'I'
        oplist.append([pauliString(ps, coeff=1.0)])
    return oplist

# w >= 2, h >= 2
def gene_dot_2d(w, h, offset=0, numq=0, interaction='Z'):
    if numq == 0:
        nq = (w+1)*(h+1)
    else:
        nq = numq
    oplist = []
    for i in range(w):
        for j in range(h):
            k = (w+1)*j + i + offset
            ps = ['I']*nq
            ps[k] = interaction
            ps[k+1] = interaction
            oplist.append([pauliString("".join(ps), coeff=1.0)])
            ps = ['I']*nq
            ps[k] = interaction
            ps[k+1+w] = interaction
            oplist.append([pauliString("".join(ps), coeff=1.0)])
    for j in range(h):
        k = (w+1)*j + w + offset
        ps = ['I']*nq
        ps[k] = interaction
        ps[k+1+w] = interaction
        oplist.append([pauliString("".join(ps), coeff=1.0)])
    for i in range(w):
        k = h*(w+1) + i + offset
        ps = ['I']*nq
        ps[k] = interaction
        ps[k+1] = interaction
        oplist.append([pauliString("".join(ps), coeff=1.0)])
    return oplist

# w, h, l >= 2
# w*(h+1)*(l+1) + (w+1)*h*(l+1) + (w+1)*(h+1)*l
def gene_dot_3d(w, h, l, interaction='Z'):
    oplist = []
    nq = (w+1)*(h+1)*(l+1)
    for i in range(l+1):
        oplist += gene_dot_2d(w, h, numq=nq, offset=i*(w+1)*(h+1), interaction=interaction)
    for i in range(l):
        for j in range((w+1)*(h+1)):
            k = i*(w+1)*(h+1) + j
            ps = ['I']*nq
            ps[k] = interaction
            ps[k+(w+1)*(h+1)] = interaction
            oplist.append([pauliString("".join(ps), coeff=1.0)])
    return oplist

if __name__ == '__main__':
    # parr = gene_dot_2d(2,2)
    # parr = gene_dot_3d(2,2,2)
    from myutil import count, test_func
    from parallel_bl import parallel_order_size_bl
    # parr = gene_dot_1d(3)
    parr = gene_dot_1d(29)
    # count(parr)
    a, b = parallel_order_size_bl(parr)
    print(a)
    # parr = gene_dot_2d(4, 5)
    # count(parr)
    # test_func(parr)
    # parr = gene_dot_3d(1,2,4)
    # count(parr)
    # test_func(parr)
    # parr = gene_dot_1d(99)
    # count(parr)
    # test_func(parr)
    # parr = gene_dot_2d(9, 9)
    # count(parr)
    # test_func(parr)
    # parr = gene_dot_3d(3,4,4)
    # count(parr)
    # test_func(parr)
