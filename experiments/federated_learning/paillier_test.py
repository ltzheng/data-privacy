import random, sys 
import time
from gmpy2 import mpz, powmod, invert, is_prime, random_state, mpz_urandomb, rint_round, log2, gcd 

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
    """
        parameters: public key, plaintext
    """
    def generate_r(n):
        """generate a random number s.t. gcd(r, n) = 1"""    
        while True:
            r = mpz(random.randint(1, n-1))
            if gcd(r, n) == 1:
                return r
    # (a * b) mod c = (a mod c * b mod c) mod c
    # c = (g^m * r^n) mod n^2 = (g^m mod n^2 * r^n mod n^2) mod n^2
    g, n = pub.g, pub.n
    r = generate_r(n)
    cipher = powmod(powmod(g, plain, n**2) * powmod(r, n, n**2), 1, n**2)
    return cipher

def dec(priv, pub, cipher):
    """
        parameters: private key, public key, cipher
    """
    n = pub.n
    x = powmod(cipher, priv.l, n**2)
    L = mpz(rint_round((x - 1) / n) - 1)
    plain = powmod(L * priv.m, 1, n)
    return plain

def enc_add(pub, m1, m2):
    """add one encrypted integer to another"""
    return powmod(enc(pub, m1) * enc(pub, m2), 1, pub.n**2)

def enc_add_const(pub, m, c):
    """add a constant to an encrypted integer"""
    return powmod(enc(pub, m) * powmod(pub.g, c, 1), 1, pub.n**2)

def enc_mul_const(pub, m, c):
    """multiply an encrypted integer by a constant"""
    return powmod(enc(pub, m), c, pub.n**2)

def test_enc_add(bit_len, priv, pub):
    elapsed_times = {}
    print('=====test enc_add=====')
    m1, m2 = mpz_urandomb(rand, bit_len - 1), mpz_urandomb(rand, bit_len - 1)
    enc_start = time.time()
    enc_plain = enc_add(pub, m1, m2)
    enc_end = time.time()
    dec_start = time.time()
    dec_cipher = dec(priv, pub, enc_plain)
    dec_end = time.time()
    print('decrypted:', dec_cipher)
    print('ground truth:', powmod(m1 + m2, 1, pub.n))
    elapsed_times['enc'] = enc_end - enc_start
    elapsed_times['dec'] = dec_end - dec_start
    print('elapsed time of encryption:', elapsed_times['enc'])
    print('elapsed time of decryption:', elapsed_times['dec'])
    return elapsed_times

def test_enc_add_const(bit_len, priv, pub):
    elapsed_times = {}
    print('=====test enc_add_const=====')
    m1, c = mpz_urandomb(rand, bit_len - 1), mpz_urandomb(rand, bit_len - 1)
    enc_start = time.time()
    enc_plain = enc_add_const(pub, m1, c)
    enc_end = time.time()
    dec_start = time.time()
    dec_cipher = dec(priv, pub, enc_plain)
    dec_end = time.time()
    print('decrypted:', dec_cipher)
    print('ground truth:', powmod(m1 + c, 1, pub.n))
    elapsed_times['enc'] = enc_end - enc_start
    elapsed_times['dec'] = dec_end - dec_start
    print('elapsed time of encryption:', elapsed_times['enc'])
    print('elapsed time of decryption:', elapsed_times['dec'])
    return elapsed_times

def test_enc_mul_const(bit_len, priv, pub):
    elapsed_times = {}
    print('=====test enc_mul_const=====')
    m1, k = mpz_urandomb(rand, bit_len - 1), mpz_urandomb(rand, bit_len - 1)
    enc_start = time.time()
    enc_plain = enc_mul_const(pub, m1, k)
    enc_end = time.time()
    dec_start = time.time()
    dec_cipher = dec(priv, pub, enc_plain)
    dec_end = time.time()
    print('decrypted:', dec_cipher)
    print('ground truth:', powmod(k * m1, 1, pub.n))
    elapsed_times['enc'] = enc_end - enc_start
    elapsed_times['dec'] = dec_end - dec_start
    print('elapsed time of encryption:', elapsed_times['enc'])
    print('elapsed time of decryption:', elapsed_times['dec'])
    return elapsed_times

if __name__ == '__main__':
    priv, pub = generate_keypair(1024)
    for bit_len in range(10, 1000, 10):
        elapsed_times = {}
        elapsed_times['enc_add'] = test_enc_add(bit_len, priv, pub)
        elapsed_times['enc_add_const'] = test_enc_add_const(bit_len, priv, pub)
        elapsed_times['enc_mul_const'] = test_enc_mul_const(bit_len, priv, pub)
        print(elapsed_times)
