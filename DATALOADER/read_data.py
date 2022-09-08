# unfortunately more number of covnolutional layers, filters and filters lenght
# don't give better accuracy
import numpy as np
import os
import sys

sys.path.append(os.getcwd())


# Set output mode:
doubles = 1
singles = 0
validation = 1

todo = [938, 783, 788, 907, 791]

# Analyse descriptor structure
desc = np.genfromtxt(r'DATA/desc.csv', dtype=str)
c_d = len(desc[0]) - 1  
site = np.genfromtxt(r'DATA/site_desc.csv', dtype=str)
c_s = len(site[0]) - 2 
c_m = 2 * c_d + c_s  
c_i = 2 * c_m 

aminos = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'HSD', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
          'THR', 'TRP', 'TYR', 'VAL']


# define some helpful functions
def find_desc(inp):
    """
    Returns an array of descriptors for a given three letter code of an aminoacid
    """
    for d in desc:
        if d[0] == inp.lower():
            out = d[1:]
            return out.astype(float)
    print("Error: Unknown code for amino acid: {0}".format(inp))
    exit()


def find_site(inp):
    """
    Returns an array of descriptors for a given mutation site
    """
    for d in site:
        if d[0] == inp:
            out = d[2:]
            return out.astype(float)
    print("Error: Unknown mutation site: {0}".format(inp))
    exit()


def mutant(inp):
    """
    Returns an array with descriptors for a given mutant XXXZZZYYY
    XXX: three letter code of wild type
    YYY: position number of the amino acid
    ZZZ: three letter code of the mutant
    Output will be: Descriptors of wildtype, position, descriptors of mutant
    """
    out = np.append(find_desc(inp[0:c_d]), find_site(inp[c_d:(c_d + c_s)]))
    out = np.append(out, find_desc(inp[(c_d + c_s):c_m]))
    return out


def single_mut(inp):
    """
    Returns a tuple of arrays with descriptors for a single mutant in double mutant style
    """
    tmp1 = mutant(inp)
    tmp2 = np.zeros(9)
    tmp2[c_d:(c_d + c_s)] = 4.0
    outp1 = np.append(tmp1, tmp2)
    outp2 = np.append(tmp2, tmp1)
    return outp1, outp2


def double_mut(inp1, inp2):
    """
    Returns a tuple of arrays with descriptors for a double mutant
    """
    mut1 = mutant(inp1)
    mut2 = mutant(inp2)
    out = np.empty((2, 18))
    out[0] = np.append(mut1, mut2)
    out[1] = np.append(mut2, mut1)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    return out


def siteall(inp):
    """
    Calculate all possible single mutations for a certain mutation site given as input
    """
    for s in site:
        if s[0] == str(inp):
            buf = s[1] + str(inp)
            break
    out = np.empty(len(aminos), dtype='<U9')

    for r in range(len(aminos)):
        out[r] = str(buf) + str(aminos[r])

    return out


# Read in barriers of single mutants and arrange input and output arrays:
barr = np.genfromtxt(r"DATA/rel.csv", dtype=str)  
inp = np.empty((0, c_i), dtype=float)
outp = np.empty(0, dtype=float)
names = np.empty((0, 2), dtype=str)
v_inp = np.empty((0, c_i), dtype=float)
v_outp = np.empty(0, dtype=float)
v_names = np.empty((0, 2), dtype=str)

###########################################################################
for n in range(len(barr[0]) - 1):
    for m in range(len(barr) - 1):
        tmp = single_mut(barr[0][n + 1] + barr[m + 1][0])
        inp = np.append(inp, tmp, axis=0)
        outp = np.append(outp, float(barr[m + 1][n + 1]))
        outp = np.append(outp, float(barr[m + 1][n + 1]))
        names = np.append(names, barr[0][n + 1] + barr[m + 1][0])
        names = np.append(names, barr[0][n + 1] + barr[m + 1][0])
idx = np.arange(len(inp))
np.random.shuffle(idx)
inp = inp[idx]
outp = outp[idx]
names = names[idx]
if singles and validation:
    cut = 300
    v_inp = inp[cut:]
    v_outp = outp[cut:]
    v_names = names[cut:]
    inp = inp[:cut]
    outp = outp[:cut]
    names = names[:cut]
############################################################################
print("Training set:   {0:3d} ins with {1:3d} outs\nValidation set: {2:3d} ins with {3:3d} outs"
      .format(len(inp),
              len(outp),
              len(v_inp),
              len(v_outp)))

# Add already explicitely tested single mutants:
print("\n===============  Processing single data  ===============\n")
sing = np.genfromtxt(r"DATA/singles.csv", dtype=str)
for b in sing:
    inp = np.append(inp, single_mut(b[0]), axis=0)
    outp = np.append(outp, float(b[1]) - 30.10)
    outp = np.append(outp, float(b[1]) - 30.10)

# Add already tested double mutants:
print("\n===============  Processing double data  ===============\n")
doub = np.genfromtxt(r"DATA/doubles.csv", dtype=str, delimiter=",")
if validation and doubles:
    v_inp = np.empty((0, c_i), dtype=float)
    v_outp = np.empty(0, dtype=float)
    v_names = np.empty((0, 2), dtype=str)
    for b in doub:
        temp = np.squeeze(double_mut(b[0], b[1]))
        v_inp = np.append(v_inp, temp, axis=0)
        v_outp = np.append(v_outp, float(b[2]) - 30.10)
        v_outp = np.append(v_outp, float(b[2]) - 30.10)
        v_names = np.append(v_names, b[0])
        v_names = np.append(v_names, b[1])
else:
    for b in doub:
        inp = np.append(inp, double_mut(b[0], b[1]), axis=0)
        outp = np.append(outp, float(b[2]) - 30.10)
        outp = np.append(outp, float(b[2]) - 30.10)

x_train, y_train = inp, outp
x_test, y_test = v_inp, v_outp
