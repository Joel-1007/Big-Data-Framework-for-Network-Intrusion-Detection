import time
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM

# Generate a random payload (simulating a network packet)
PAYLOAD_SIZE = 1024 * 1024  # 1MB Packet
payload = os.urandom(PAYLOAD_SIZE)
key_32 = os.urandom(32) # 256-bit key
nonce_12 = os.urandom(12)

print(f"--- CRYPTO BENCHMARK (Packet Size: {PAYLOAD_SIZE/1024:.0f} KB) ---")

# 1. Benchmark ChaCha20-Poly1305
start = time.time()
chacha = ChaCha20Poly1305(key_32)
for _ in range(1000):
    ct = chacha.encrypt(nonce_12, payload, None)
    pt = chacha.decrypt(nonce_12, ct, None)
end = time.time()
chacha_time = end - start
print(f"ChaCha20-Poly1305 Time: {chacha_time:.4f}s")

# 2. Benchmark AES-256-GCM
start = time.time()
aes = AESGCM(key_32)
for _ in range(1000):
    ct = aes.encrypt(nonce_12, payload, None)
    pt = aes.decrypt(nonce_12, ct, None)
end = time.time()
aes_time = end - start
print(f"AES-256-GCM Time:       {aes_time:.4f}s")

# 3. Conclusion
print("-" * 40)
if chacha_time < aes_time:
    print(f"WINNER: ChaCha20 is {(aes_time/chacha_time):.2f}x FASTER in this environment.")
    print("REASON: Ideal for software-based containers (Docker) without HW acceleration.")
else:
    print(f"WINNER: AES-256 is {(chacha_time/aes_time):.2f}x FASTER (HW Acceleration detected).")
    print("REASON: CPU has AES-NI instructions enabled.")
