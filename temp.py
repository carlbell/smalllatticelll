from hsnf import column_style_hermite_normal_form, row_style_hermite_normal_form
from numpy.random import default_rng
import scipy
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-n', default=2, type=int)
parser.add_argument('-m', default=4, type=int)
parser.add_argument('-q', default=7, type=int)
parser.add_argument('-bases_out', default="bases_out.txt")
parser.add_argument('-hermite_out', default="hermite_out.txt")
parser.add_argument('--write_bases', default=False, action='store_true')
parser.add_argument('--write_hermite', default=False, action='store_true')
parser.add_argument('--read_bases', default=False, action='store_true')
args = parser.parse_args()


def strip_zero_rows(basis):
    return basis[~np.all(basis == 0, axis=1)]

def get_basis(n, m, q, rng):
    A = rng.integers(low=0, high=q, size=(n,m))
    print(A)

    c = np.concatenate([np.ones(n), np.zeros(m)])
    temp = -q * np.identity(n)
    temp = temp.astype(int)
    #print(temp)
    Aeq = np.concatenate([temp, A], axis=1)
    #print(Aeq)
    beq = np.zeros(n)
    mybounds = [(0,None)] * n + [(0,q-1)] * m



            
    Aub = -1*np.ones(shape=(1,n+m))
    bub = -1*np.ones(1)
    solutions = []
    solution = scipy.optimize.linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=mybounds, method='highs', callback=None, options=None, x0=None, integrality=np.ones(n+m))
    prev_x = solution['x']
    #print("\n\nsolution", prev_x)
    #print("Aeq", Aeq)
    #print("beq", beq)
    #print("Aeq*solution", np.matmul(Aeq, prev_x))



    prev_x = np.concatenate([np.zeros(n), prev_x[n:]])
    solutions.append(prev_x[n:])
    Aub = np.append(Aub, [prev_x], axis=0)
    bub = np.append(bub, [sum(prev_x)])
    for i in range(10):
        #print(Aub)
        solution = scipy.optimize.linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=mybounds, method='highs', callback=None, options=None, x0=None, integrality=np.ones(n+m))
        prev_x = solution['x']
        if prev_x is None:
            break
        #print("\n\nsolution", prev_x)
        #print("Aeq", Aeq)
        #print("beq", beq)
        #print("Aeq*solution", np.matmul(Aeq, prev_x))
        prev_x = np.concatenate([np.zeros(n), prev_x[n:]])
        solutions.append(prev_x[n:])
        #print(prev_x)
        Aub = np.append(Aub, [prev_x], axis=0)
        bub = np.append(bub, [sum(prev_x)-1])
        if solutions[-1] is None:
              solutions = solutions[:-1]

    solutions = np.array(solutions).astype(int)
    print(solutions)
    return solutions


def proj(u,v):
    return (np.dot(u,v)/np.dot(u,u))*u

def gram_schmidt(X):
    Y = []
    for (i, inX) in enumerate(X):
        temp_vec = inX
        for inY in Y:
            if not np.any(inY):
                break
            proj_vec = proj(inY, inX)
            print("i =", i, ", projection vector =", proj_vec)
            temp_vec = temp_vec - proj_vec
            print("i =", i, ", temporary vector =", temp_vec)
        Y.append(temp_vec)
    return Y

#print(gram_schmidt(solutions))
#H, R = column_style_hermite_normal_form(solutions)
#print(H)

#H, L = row_style_hermite_normal_form(solutions)

#print(H)
#print(L)

# gram schmidt and LLL stuff
def projection_scale(u, v):
    '''Computes <u,v>/<u,u>, which is the scale used in projection.'''
    return np.dot(u, v) / np.dot(u, u)

def proj(u, v):
    '''Computes the projection of vector v onto vector u. Assumes u is not zero.'''
    return np.dot(projection_scale(u, v), u)

def gram_schmidt(basis, orthobasis): 
    '''Computes Gram Schmidt orthoganalization (without normalization) of a basis.'''
    orthobasis[0] = basis[0]
    for i in range(1, basis.shape[1]):  # Loop through dimension of basis.
        orthobasis[i] = basis[i]
        for j in range(0, i):
            orthobasis[i] -= proj(orthobasis[j], basis[i])
    return orthobasis

def reduction(basis, orthobasis):
    '''Performs length reduction on a basis.'''
    total_reduction = 0 # Track the total amount by which the working vector is reduced.
    for j in range(k-1, -1, -1):   # j loop. Loop down from k-1 to 0.
        m = round(projection_scale(orthobasis[j], basis[k]))
        total_reduction += np.dot(m, basis[j])[0]
        basis[k] -= np.dot(m, basis[j]) # Reduce the working vector by multiples of preceding vectors.
    if total_reduction > 0:
        gram_schmidt(basis, orthobasis)

def lovasz(basis, orthobasis):
    '''Checks the Lovasz condition for a basis. Either swaps adjacent basis vectors and recomputes Gram-Scmidt or increments the working index.'''
    c = DELTA - projection_scale(orthobasis[k-1], basis[k])**2
    if la.norm(orthobasis[k])**2 >= np.dot(c, la.norm(orthobasis[k-1]**2)): # Check the Lovasz condition.
        k += 1  # Increment k if the condition is met.
    else: 
        basis[[k, k-1]] = basis[[k-1, k]] # If the condition is not met, swap the working vector and the immediately preceding basis vector.
        gram_schmidt() # Recompute Gram-Schmidt if swap
        k = max([k-1, 1])

def LLL(basis):
    orthobasis = basis.copy()
    global DELTA
    DELTA = 0.91
    global k
    k = 1
    gram_schmidt(basis, orthobasis)
    steps = 0
    while k <= basis.shape[1] - 1:
        reduction(basis, orthobasis)
        steps += 1
        print('\\begin{block}{Step ', steps, 'of LLL,', 'reduction step}\\begin{center} $M=', bmatrix(basis.transpose()),'$ \\end{center}\\end{block}')
        lovasz(basis, orthobasis)
        steps +=1
        print('\\begin{block}{Step ', steps, 'of LLL.', 'checking Lovasz condition}\\begin{center} $M=', bmatrix(basis.transpose()),'$ \\end{center}\\end{block}')
    print('LLL Reduced Basis:\n', basis.transpose())

# keep creating random bases, note ||b_1||/det(L)




if args.write_bases:
    with open(args.bases_out, "wb") as bases_out:      
        rng = default_rng()
        
        n = args.n
        m = args.m
        q = args.q
        for i in range(10):
            possible_basis = get_basis(n, m, q, rng)
            possible_basis = strip_zero_rows(possible_basis)
            if len(possible_basis) == n:
                np.save(bases_out, possible_basis)

