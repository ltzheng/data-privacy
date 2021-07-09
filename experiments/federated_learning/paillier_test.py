import random, sys 
import time
import numpy as np
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd 
from termcolor import colored

rand = random_state(random.randrange(sys.maxsize))

# private key: (lambda, mu)
class PrivateKey(object):
    def __init__(self, p, q, n):
        # Euler function lambda
        if p == q:
            self.l = p * (p - 1)
        else:
            self.l = (p - 1) * (q - 1)
        try:
            self.m = invert(self.l, n)  # mu
        except ZeroDivisionError as e:
            print(e)
            exit()

# public key: (n, g)
class PublicKey(object):
    def __init__(self, n):
        self.n = n
        self.n_sq = n * n
        self.g = n + 1
        self.bits = mpz(rint_round(log2(self.n)))

def generate_prime(bits):    
    """generate an b-bit prime integer"""    
    while True:
        possible = mpz(2)**(bits - 1) + mpz_urandomb(rand, bits - 1)
        if is_prime(possible):
            return possible

def generate_keypair(bits):
    """generate a pair of paillier keys with bits > 5"""
    p = generate_prime(bits // 2)
    q = generate_prime(bits // 2)
    n = p * q
    return PrivateKey(p, q, n), PublicKey(n)

def enc(pub, plain):
    """Parameters: public key, plaintext"""

    def generate_r(n):
        """generate a random number s.t. gcd(r, n) = 1"""    
        while True:
            r = mpz(random.randint(1, n - 1))
            if gcd(r, n) == 1:
                return r
    
    # Mathematically:
    # (a * b) mod c = (a mod c * b mod c) mod c
    # c = (g^m * r^n) mod n^2 = (g^m mod n^2 * r^n mod n^2) mod n^2
    g, n, n_sq = pub.g, pub.n, pub.n_sq
    r = generate_r(n)
    cipher = powmod(powmod(g, plain, n_sq) * powmod(r, n, n_sq), 1, n_sq)
    return cipher

def dec(priv, pub, cipher):
    """Parameters: private key, public key, cipher"""
    n, n_sq = pub.n, pub.n_sq
    x = powmod(cipher, priv.l, n_sq)
    L = np.floor((x - 1) // n)
    plain = powmod(mpz(L * priv.m), 1, n)
    return plain

def enc_add(pub, m1, m2):
    """Add one encrypted integer to another"""
    return powmod(m1 * m2, 1, pub.n_sq)

def enc_add_const(pub, m, c):
    """Add a constant to an encrypted integer"""
    n_sq = pub.n_sq
    return powmod(powmod(m, 1, n_sq) * powmod(pub.g, c, n_sq), 1, n_sq)

def enc_mul_const(pub, m, c):
    """Multiply an encrypted integer by a constant"""
    return powmod(m, c, pub.n_sq)

def test(mode, bit_len, priv, pub):
    def generate_num(bit_len):
        return mpz(2)**(bit_len - 1) + mpz_urandomb(rand, bit_len - 1)
    
    elapsed_times = {}
    print('=====TEST ' + mode + '=====')
    a = generate_num(bit_len)
    b = generate_num(bit_len)
    c = generate_num(bit_len)
    m1 = enc(pub, a)
    m2 = enc(pub, b)

    enc_start = time.time()
    if mode == 'enc_add':
        enc_plain = enc_add(pub, m1, m2)
        ground_truth = powmod(a + b, 1, pub.n)
    elif mode == 'enc_add_const':
        enc_plain = enc_add_const(pub, m1, c)
        ground_truth = powmod(a + c, 1, pub.n)
    elif mode == 'enc_mul_const':
        enc_plain = enc_mul_const(pub, m1, c)
        ground_truth = powmod(a * c, 1, pub.n)
    else:
        raise NotImplementedError
    enc_end = time.time()

    dec_start = time.time()
    dec_cipher = dec(priv, pub, enc_plain)
    dec_end = time.time()
    if dec_cipher == ground_truth:
        print(colored('PASS', 'green'))
    else:
        print(colored('FAIL', 'red'))
    
    elapsed_times['enc'] = enc_end - enc_start
    elapsed_times['dec'] = dec_end - dec_start
    print('elapsed time of encryption:', elapsed_times['enc'])
    print('elapsed time of decryption:', elapsed_times['dec'])
    return elapsed_times


if __name__ == '__main__':
    priv, pub = generate_keypair(1024)
    for bit_len in range(10, 1000, 10):
        elapsed_times = {}
        elapsed_times['enc_add'] = test('enc_add', bit_len, priv, pub)
        elapsed_times['enc_add_const'] = test('enc_add_const', bit_len, priv, pub)
        elapsed_times['enc_mul_const'] = test('enc_mul_const', bit_len, priv, pub)
        print(elapsed_times)
