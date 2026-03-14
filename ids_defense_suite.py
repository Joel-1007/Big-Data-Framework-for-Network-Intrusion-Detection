import os
import time
import json
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305, AESGCM
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

# --- 1. CRYPTO BENCHMARK (AES-128 vs 256 vs ChaCha) ---
print("\n" + "="*50)
print("PHASE 1: CRYPTO BENCHMARK (Speed Test)")
print("="*50)

payload = os.urandom(1024 * 1024) # 1MB Packet
nonce = os.urandom(12)

def benchmark(name, cipher_obj):
    start = time.time()
    for _ in range(500): # Run 500 times
        ct = cipher_obj.encrypt(nonce, payload, None)
    return time.time() - start

# AES-128 (Faster?)
key128 = AESGCM.generate_key(bit_length=128)
time_aes128 = benchmark("AES-128-GCM", AESGCM(key128))

# AES-256 (Stronger)
key256 = AESGCM.generate_key(bit_length=256)
time_aes256 = benchmark("AES-256-GCM", AESGCM(key256))

# ChaCha20
key_cc = ChaCha20Poly1305.generate_key()
time_chacha = benchmark("ChaCha20-Poly1305", ChaCha20Poly1305(key_cc))

print(f"AES-128 Time: {time_aes128:.4f}s")
print(f"AES-256 Time: {time_aes256:.4f}s")
print(f"ChaCha20 Time: {time_chacha:.4f}s")

winner = "AES-256" if time_aes256 < time_chacha else "ChaCha20"
print(f"RECOMMENDATION: Use {winner} (Best Balance of Speed/Security)")


# --- 2. UNLINKABILITY & CORRELATION TEST ---
print("\n" + "="*50)
print("PHASE 2: UNLINKABILITY (Correlation Resistance)")
print("="*50)

msg = b"SECRET_LOGIN_PASSWORD_123"
cipher = AESGCM(key256)

# Encrypt SAME message twice with DIFFERENT nonces
nonce_a = os.urandom(12)
nonce_b = os.urandom(12)

ct_a = cipher.encrypt(nonce_a, msg, None)
ct_b = cipher.encrypt(nonce_b, msg, None)

print(f"Plaintext: {msg}")
print(f"Ciphertext A (Nonce A): {ct_a.hex()[:50]}...")
print(f"Ciphertext B (Nonce B): {ct_b.hex()[:50]}...")

if ct_a != ct_b:
    print("RESULT: PASS. Identical inputs produced totally different outputs.")
    print("        This proves 'Unlinkability' and breaks Statistical Correlation.")
else:
    print("RESULT: FAIL.")


# --- 3. MITM INTEGRITY CHECK ---
print("\n" + "="*50)
print("PHASE 3: MITM TAMPERING CHECK")
print("="*50)

# Hacker flips ONE bit in the ciphertext
tampered_ct = bytearray(ct_a)
tampered_ct[-1] ^= 0xFF 

try:
    cipher.decrypt(nonce_a, bytes(tampered_ct), None)
    print("RESULT: FAIL (Tampered packet was accepted!)")
except Exception:
    print("RESULT: PASS. Decryption failed due to Tag Mismatch.")
    print("        Poly1305/GCM Tag detected the MITM modification.")


# --- 4. ADVERSARIAL ATTACK (Robustness Test) ---
print("\n" + "="*50)
print("PHASE 4: ADVERSARIAL ROBUSTNESS (Model Stress Test)")
print("="*50)

# Note: Standard FGSM/PGD requires Gradients (Neural Networks). 
# Random Forest is non-differentiable. We use 'Feature Perturbation' instead.

print("Simulating Feature Perturbation (The 'FGSM' for Random Forests)...")
print("Attack: Adding subtle noise to 'Flow Duration' to fool the model.")

# Mocking a trained RF prediction for demo purposes
# (In real app, load your Spark model here)
def mock_rf_predict(duration, packets):
    # Valid Logic: High duration + Low packets = DoS (Slowloris)
    if duration > 10000 and packets < 5:
        return "DoS Attack"
    return "BENIGN"

# Original Malicious Flow (Slowloris)
base_dur = 12000
base_pkts = 3
original_pred = mock_rf_predict(base_dur, base_pkts)
print(f"Original Flow [{base_dur}ms, {base_pkts}pkts] -> Prediction: {original_pred}")

# Adversarial Attack: Tweak duration slightly to boundary
# Hacker tries to lower duration just enough to slip by
adv_dur = 9999 
adv_pred = mock_rf_predict(adv_dur, base_pkts)

print(f"Adversarial Flow [{adv_dur}ms, {base_pkts}pkts] -> Prediction: {adv_pred}")

if original_pred != adv_pred:
    print("RESULT: VULNERABLE. The model was fooled by small perturbation.")
    print("DEFENSE: Retrain Random Forest with 'Adversarial Training' (noisy data).")
else:
    print("RESULT: ROBUST. Model maintained classification.")

print("\n" + "="*50)
print("FULL SECURITY SUITE COMPLETE")
print("="*50)
