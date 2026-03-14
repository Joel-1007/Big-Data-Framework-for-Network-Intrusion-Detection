import os
import json
import time
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# --- CONFIGURATION ---
# We use ChaCha20-Poly1305 (Modern, Fast, Secure)
# Key must be 32 bytes (256-bit)
SECRET_KEY = os.urandom(32)
cipher = ChaCha20Poly1305(SECRET_KEY)

print("--- HYBRID DEFENSE SYSTEM INITIALIZED ---")
print(f"Encryption Engine: ChaCha20-Poly1305 | Key: [HIDDEN_32_BYTES]")

# Database of seen Nonces (to prevent Replay Attacks)
seen_nonces = set()

def encrypt_packet(data_dict):
    """Simulates a client sending an encrypted packet with a unique Nonce."""
    nonce = os.urandom(12) # Unique 12-byte nonce
    plaintext = json.dumps(data_dict).encode('utf-8')
    ciphertext = cipher.encrypt(nonce, plaintext, None)
    return nonce, ciphertext

def decrypt_and_check(nonce, ciphertext):
    """Layer 1: Encryption & Integrity Check (Stops MITM / Replay)"""
    
    # 1. Replay Check
    if nonce in seen_nonces:
        return None, "BLOCKED: Replay Attack Detected (Nonce reused)"
    seen_nonces.add(nonce)
    
    # 2. Decryption (Integrity Check)
    try:
        plaintext = cipher.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext), "PASS: Integrity OK"
    except Exception:
        return None, "BLOCKED: Decryption Failed (MITM Tampering)"

# --- SIMULATION ---

# 1. A Valid User Packet (Normal Traffic)
valid_user_data = {
    "Source IP": "192.168.1.5",
    "Destination Port": 443,
    "Flow Duration": 500,
    "Label": "BENIGN"
}

print("\n--- SCENARIO 1: Valid User Traffic ---")
nonce1, encrypted1 = encrypt_packet(valid_user_data)
data, status = decrypt_and_check(nonce1, encrypted1)
print(f"Packet 1: {status}")

# 2. The Hacker Replays the SAME packet (Replay Attack)
print("\n--- SCENARIO 2: Replay Attack (Hacker copies Packet 1) ---")
# Hacker tries to send the exact same nonce and ciphertext
data_replay, status_replay = decrypt_and_check(nonce1, encrypted1)
print(f"Packet 1 (Replay): {status_replay}")
if status_replay.startswith("BLOCKED"):
    print(">> SUCCESS: Encryption Layer stopped the attack before ML even saw it.")

# 3. Adversarial/Tampering (MITM)
print("\n--- SCENARIO 3: MITM Attack (Hacker changes payload) ---")
# Hacker captures packet, flips a bit, and sends it
tampered_ciphertext = bytearray(encrypted1)
tampered_ciphertext[-1] ^= 0x01 # Flip last bit
data_mitm, status_mitm = decrypt_and_check(nonce1, bytes(tampered_ciphertext))
print(f"Packet 1 (Modified): {status_mitm}")

# 4. DoS Attack (ML Layer Needed!)
print("\n--- SCENARIO 4: DoS Attack (Fresh Packets) ---")
print("Hacker generates NEW encrypted packets (bypassing Replay Check)...")
print("Passing to Layer 2: Machine Learning...")

# Load your Spark ML Model (Simulation logic for demo)
# In real life: model = PipelineModel.load("rf_model")
def ml_predict(flow_data):
    # Simulating the ML prediction based on flow features
    if flow_data['Flow Duration'] > 10000:
        return "ALERT: DoS Attack Detected by Random Forest"
    return "Benign"

# Hacker sends a massive flow (valid crypto, but malicious behavior)
dos_data = {
    "Source IP": "205.174.165.73",
    "Destination Port": 80,
    "Flow Duration": 50000, # Suspiciously long
    "Label": "DoS Hulk"
}

nonce2, encrypted2 = encrypt_packet(dos_data)
# Layer 1: Crypto Check
payload, status = decrypt_and_check(nonce2, encrypted2)
if "PASS" in status:
    # Layer 2: ML Check
    ml_verdict = ml_predict(payload)
    print(f"Layer 1 Status: {status}")
    print(f"Layer 2 Verdict: {ml_verdict}")

print("\n--- HYBRID SYSTEM SUMMARY ---")
print("1. Replay Attacks  -> Stopped by ChaCha20 Nonce Check")
print("2. MITM Tampering  -> Stopped by Poly1305 Integrity Check")
print("3. DoS / Botnets   -> Stopped by Random Forest ML Model")
