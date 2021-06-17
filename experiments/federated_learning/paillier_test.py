"""
"""
import random, sys 

from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd 

rand=random_state(random.randrange(sys.maxsize))

class PrivateKey(object):
    def __init__(self, p, q, n):
        if p==q:
            self.l = p * (p-1)
        else:
            self.l = (p-1) * (q-1)
        try:
            self.m = invert(self.l, n)
        except ZeroDivisionError as e:
            print(e)
            exit()

class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits=mpz(rint_round(log2(self.n)))

def generate_prime(bits):    
    """Will generate an integer of b bits that is prime using the gmpy2 library  """    
    while True:
        possible =  mpz(2)**(bits-1) + mpz_urandomb(rand, bits-1 )
        if is_prime(possible):
            return possible

def generate_keypair(bits):
    """ Will generate a pair of paillier keys bits>5"""
    p = generate_prime(bits // 2)
    #print(p)
    q = generate_prime(bits // 2)
    #print(q)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)

def enc(pub, plain):#(public key, plaintext) #to do
    return cipher

def dec(priv, pub, cipher): #(private key, public key, cipher) #to do
    return plain

def enc_add(pub, m1, m2): #to do
    """Add one encrypted integer to another"""
    return 

def enc_add_const(pub, m, c): #to do
    """Add constant n to an encrypted integer"""
    return

def enc_mul_const(pub, m, c): #to do
    """Multiplies an encrypted integer by a constant"""
    return

if __name__ == '__main__':
    priv, pub = generate_keypair(1024)
    """
    test
    """

