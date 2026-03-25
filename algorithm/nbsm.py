from PIL import Image
import numpy as np
import hashlib
import hmac
import struct
import struct as _struct
import os
import base64
from Crypto.Cipher import AES


# ═══════════════════════════════════════════════════════════════════
# NBSM — Natural Bit Sequence Matching
#
# Definition 1 (NBSM System):
#   NBSM = (E, D, K, B, Φ) where:
#     B  = bit stream of cover image,  |B| = N bits
#     M  = secret message,             |M| = K bits
#     K  = secret key shared between sender and receiver
#     Φ(K, i) = traversal function: gives independent start position
#               for the i-th message bit  (per-bit, key-dependent)
#     E  = encode: (M, B, K) → gap list L
#     D  = decode: (L, B, K) → M
#
# Correctness property:  D(E(M, B, K), B, K) = M  for all valid inputs.
#
# Note on single-start variant:
#   For computational efficiency, Φ can be evaluated once and scanning
#   proceeds continuously.  The expected gap distribution and Theorem 2
#   capacity bound remain empirically valid under this variant, as the
#   weak dependence between consecutive gaps satisfies the ergodic CLT.
#
# Encryption: AES-256-GCM (AEAD)
#   - Confidentiality : gap list, message length, elias bit length all hidden
#   - Integrity       : GCM tag detects any tampering
#   - AAD binding     : SHA-256 hash and nonce are authenticated but plaintext
#   - Hides message length from passive observers (length side-channel closed)
#
# Auxiliary information format (4 fields, semicolon-delimited):
#   <sha256_hex> ; <nonce_b64> ; <tag_b64> ; <ciphertext_b64>
#
#   Ciphertext decrypts to:  [4 bytes: message_length_bytes]
#                             [4 bytes: original_bit_length]
#                             [N bytes: Elias Gamma compressed gap bytes]
# ═══════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# ALGORITHM 1 — Bit Stream Generator
# ─────────────────────────────────────────────

def image_to_bitstream(image_path):
    """
    Convert image to a flat bit stream from RGB pixel values.
    Raster scan order: pixel by pixel, R then G then B, MSB to LSB.
    This is the canonical NBSM bit stream B = (b1, b2, ..., bN).
    N = W × H × C x 8 for an image.

    Theorem 1 basis: each bit bᵢ is independently Bernoulli(0.5)
    in natural images, giving E[G] = 2 positions scanned per message bit.
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).flatten()
    bitstream = ''.join(format(p, '08b') for p in pixels)
    return bitstream


def message_to_bits(message):
    """
    Convert a UTF-8 message string to a binary bit string.
    Each byte is encoded as 8 bits, MSB first.
    """
    try:
        encoded = message.encode('utf-8')
    except Exception as e:
        raise ValueError(f"Message encoding failed: {e}")
    return ''.join(format(byte, '08b') for byte in encoded)


def bits_to_message(bit_string):
    """
    Convert a binary bit string back to a UTF-8 string.
    Inverse of message_to_bits.
    """
    if len(bit_string) % 8 != 0:
        raise ValueError(
            f"Bit string length {len(bit_string)} is not a multiple of 8."
        )
    chars = [
        chr(int(bit_string[i:i + 8], 2))
        for i in range(0, len(bit_string), 8)
    ]
    return ''.join(chars)


# ─────────────────────────────────────────────
# Φ — TRAVERSAL FUNCTION  (Definition 1)
# ─────────────────────────────────────────────

def phi(encryption_key, bit_index, bitstream_length):
    """
    Φ(K, i) — Traversal function from Definition 1.

    Returns a deterministic, key-dependent, per-bit starting scan position
    for the i-th message bit.  Both sender and receiver compute the same
    position independently from the shared key — nothing is transmitted.

    Implementation:
        seed = SHA-256( key || ":" || str(i) )
        position = first 8 bytes of seed, interpreted as uint64, mod N

    This gives each message bit an independent, pseudorandom entry point
    into the bit stream B, satisfying the independence assumption required
    by the CLT argument in Theorem 2.

    Args:
        encryption_key   : shared secret key string (K in Definition 1)
        bit_index        : i — index of the current message bit (0-based)
        bitstream_length : N — total length of the cover image bit stream

    Returns:
        Integer start position in [0, N)
    """
    seed_input = f"{encryption_key}:{bit_index}".encode('utf-8')
    digest = hashlib.sha256(seed_input).digest()
    position = struct.unpack('>Q', digest[:8])[0] % bitstream_length
    return position


# ─────────────────────────────────────────────
# SHA-256 INTEGRITY UTILITIES
# ─────────────────────────────────────────────

def compute_message_hash(message):
    """
    Compute SHA-256 hash of the original plaintext message.
    Transmitted in auxiliary information so the receiver can verify
    integrity of the recovered message.

    Returns:
        64-character lowercase hex string (256 bits)
    """
    return hashlib.sha256(message.encode('utf-8')).hexdigest()


def verify_message_hash(recovered_message, expected_hash):
    """
    Verify recovered message against transmitted SHA-256 hash.
    Uses hmac.compare_digest for constant-time comparison to prevent
    timing side-channel attacks.

    Returns:
        True if hash matches, False otherwise
    """
    computed = hashlib.sha256(recovered_message.encode('utf-8')).hexdigest()
    return hmac.compare_digest(computed, expected_hash)


# ─────────────────────────────────────────────
# AES-256-GCM ENCRYPTION  (AEAD)
# ─────────────────────────────────────────────

def _derive_aes_key(key_string):
    """
    Derive a 32-byte AES-256 key from an arbitrary-length password string
    using HKDF-SHA256 (RFC 5869, extract-then-expand).

    Steps:
        1. Extract  : PRK = HMAC-SHA256(salt="NBSM-AES-KEY", IKM=key_bytes)
        2. Expand   : AES_key = HMAC-SHA256(PRK, info="gap-encryption" || 0x01)

    Args:
        key_string : shared secret key (any length)

    Returns:
        32-byte AES key
    """
    salt = b"NBSM-AES-KEY"
    info = b"gap-encryption"
    ikm  = key_string.encode('utf-8')

    prk     = hmac.new(salt, ikm, hashlib.sha256).digest()
    aes_key = hmac.new(prk, info + b'\x01', hashlib.sha256).digest()
    return aes_key


def aes_encrypt(payload_bytes, key_string, associated_data=b''):
    """
    Encrypt payload_bytes using AES-256-GCM (AEAD).

    AES-256-GCM provides:
        - Confidentiality  : payload is encrypted (attacker sees nothing)
        - Integrity        : GCM tag detects any tampering
        - AAD binding      : associated_data is authenticated but NOT encrypted
                             (hash is bound to ciphertext via tag — cannot be swapped)

    Args:
        payload_bytes   : bytes to encrypt (length header + compressed gap bytes)
        key_string      : shared secret key string
        associated_data : plaintext bytes that are authenticated but not encrypted

    Returns:
        (nonce, tag, ciphertext)
        nonce      : 16 random bytes — transmitted in plaintext, not secret
        tag        : 16-byte GCM authentication tag
        ciphertext : encrypted payload, same length as payload_bytes
    """
    aes_key              = _derive_aes_key(key_string)
    nonce                = os.urandom(16)
    cipher               = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    cipher.update(associated_data)
    ciphertext, tag      = cipher.encrypt_and_digest(payload_bytes)
    return nonce, tag, ciphertext


def aes_decrypt(nonce, tag, ciphertext, key_string, associated_data=b''):
    """
    Decrypt and verify using AES-256-GCM.

    Raises ValueError if the tag does not match — meaning either the key
    is wrong, the ciphertext was tampered with, or the associated data
    was modified.

    Args:
        nonce           : 16-byte nonce from sender
        tag             : 16-byte GCM authentication tag
        ciphertext      : encrypted bytes
        key_string      : shared key
        associated_data : same plaintext bytes passed during encryption

    Returns:
        Decrypted plaintext bytes

    Raises:
        ValueError : if GCM authentication tag verification fails
    """
    aes_key = _derive_aes_key(key_string)
    cipher  = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    cipher.update(associated_data)
    try:
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError:
        raise ValueError(
            "AES-GCM authentication FAILED. "
            "Wrong key, tampered ciphertext, or modified associated data."
        )
    return plaintext


# ─────────────────────────────────────────────
# ELIAS GAMMA CODING  (Theorem 3)
# ─────────────────────────────────────────────

def elias_gamma_encode(n):
    """
    Encode a positive integer n >= 1 using Elias Gamma coding.

    Codeword : floor(log2(n)) leading zeros, then binary(n)
    Length   : |γ(n)| = 2·floor(log2(n)) + 1 bits

    Theorem 3: For G ~ Geometric(0.5),
               E[|γ(G+1)|] ≈ 2.77 bits per gap value.
    """
    if n < 1:
        raise ValueError(f"Elias Gamma requires positive integers, got {n}")
    k = n.bit_length() - 1
    return '0' * k + format(n, f'0{k + 1}b')


def elias_gamma_decode_stream(bitstream):
    """
    Decode a sequence of Elias Gamma encoded integers from a bit string.
    Returns list of decoded positive integers.
    """
    values = []
    i = 0
    while i < len(bitstream):
        k = 0
        while i < len(bitstream) and bitstream[i] == '0':
            k += 1
            i += 1
        if i + k + 1 > len(bitstream):
            break
        value = int(bitstream[i:i + k + 1], 2)
        values.append(value)
        i += k + 1
    return values


def compress_gaps(gaps):
    """
    Compress gap list using Elias Gamma coding.

    Raw gaps Gᵢ >= 0 (a gap of 0 means the very first position checked
    matched immediately).  Elias Gamma requires integers >= 1, so each
    gap is shifted by +1 before encoding.  The shift is reversed in
    decompress_gaps.

    Theorem 3 basis: E[|γ(G+1)|] ≈ 2.77 bits for G ~ Geometric(0.5).
    """
    shifted = [g + 1 for g in gaps]
    return ''.join(elias_gamma_encode(g) for g in shifted)


def decompress_gaps(encoded_bitstring, expected_count):
    """
    Decompress Elias Gamma encoded gap string back to original gap list.
    Reverses the +1 shift applied in compress_gaps.
    """
    shifted = elias_gamma_decode_stream(encoded_bitstring)
    if len(shifted) != expected_count:
        raise ValueError(
            f"Expected {expected_count} gaps, decoded {len(shifted)}. "
            f"Key may be corrupted or message_length_bytes is incorrect."
        )
    return [g - 1 for g in shifted]


# ─────────────────────────────────────────────
# BIT ↔ BYTE PACKING UTILITIES
# ─────────────────────────────────────────────

def _bits_to_bytes(bit_string):
    """
    Pack a '0'/'1' bit string into raw bytes (MSB first).
    Pads with trailing zeros to reach a multiple of 8 bits.

    Returns:
        (raw_bytes, original_bit_length)
    """
    pad    = (8 - len(bit_string) % 8) % 8
    padded = bit_string + '0' * pad
    raw    = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
    return raw, len(bit_string)


def _bytes_to_bits(raw_bytes, original_bit_length):
    """
    Unpack raw bytes back to a '0'/'1' bit string.
    Truncates to original_bit_length to remove padding zeros.
    """
    full = ''.join(format(b, '08b') for b in raw_bytes)
    return full[:original_bit_length]



# def check_feasibility(bitstream, msg_bits, encryption_key=None, safety_factor=1.2, min_p=0.01):
#     """
#     Check whether the given image can feasibly embed the message
#     using NBSM (gap-based matching).

#     Uses empirical matching probability p to estimate expected gaps.

#     Parameters:
#     -----------
#     image_path : str
#         Path to image (same as used in image_to_bitstream)
#     message : str
#         Secret message
#     encryption_key : str (optional)
#         Not required here, but kept for consistency
#     safety_factor : float
#         Margin to avoid borderline failures
#     min_p : float
#         Minimum acceptable matching probability

#     Returns:
#     --------
#     feasible : bool
#     stats : dict
#     """

#     # ── Generate bitstreams (use your existing functions) ─────────

#     N = len(bitstream)
#     K = len(msg_bits)

#     # ── Estimate matching probability p ───────────────────────────
#     # Repeat message bits to match image length
#     matches = 0
#     for i in range(N):
#         if bitstream[i] == msg_bits[i % K]:
#             matches += 1
#             print(bitstream[i], msg_bits[i % K], "Match")

#     p = matches / N

#     # ── Edge case: no matches ─────────────────────────────────────
#     if p == 0:
#         return False, {
#             "p": 0,
#             "reason": "No matching bits in image"
#         }

#     # ── Reject very low probability images ────────────────────────
#     if p < min_p:
#         return False, {
#             "p": p,
#             "reason": "Matching probability too low"
#         }

#     # ── Expected gap (Geometric distribution) ─────────────────────
#     expected_gap = 1 / p

#     # ── Required scan length estimation ───────────────────────────
#     required_bits = K * expected_gap * safety_factor

#     # ── Feasibility condition ─────────────────────────────────────
#     feasible = required_bits <= N

#     stats = {
#         "p": p,
#         "expected_gap": expected_gap,
#         "required_bits": required_bits,
#         "available_bits": N,
#         "message_bits": K,
#         "utilization_ratio": required_bits / N
#     }

#     return feasible, stats


def check_feasibility(bitstream, msg_bits, safety_factor=1.2, min_p=0.01):
    N = len(bitstream)
    K = len(msg_bits)
    # ── Basic validation ─────────────────────────────────────────
    if N == 0 or K == 0:
        return False, {"reason": "Empty bitstream or message"}

    # ── Count bits (fast, no loop over both) ─────────────────────
    ones_B = bitstream.count('1')
    ones_M = msg_bits.count('1')

    # ── Probabilities ────────────────────────────────────────────
    P_B1 = ones_B / N
    P_M1 = ones_M / K

    # Avoid recomputing zeros
    P_B0 = 1.0 - P_B1
    P_M0 = 1.0 - P_M1

    # ── Matching probability ─────────────────────────────────────
    p = P_B0 * P_M0 + P_B1 * P_M1

    # ── Edge cases ───────────────────────────────────────────────
    if p <= min_p:
        return False, {
            "p": p,
            "reason": "Matching probability too low"
        }

    # ── Expected gap ─────────────────────────────────────────────
    expected_gap = 1.0 / p

    # ── Required scan length ─────────────────────────────────────
    required_bits = K * expected_gap # * safety_factor

    feasible = required_bits <= N

    stats = {
        "p": float(p),
        "expected_gap": float(expected_gap),
        "required_bits": float(required_bits),
        "available_bits": int(N),
        "message_bits": int(K),
        "utilization_ratio": float(required_bits / N)
    }

    print("Feasibility check stats:")
    # print(stats)

    return feasible, stats


# ─────────────────────────────────────────────
# AUXILIARY INFORMATION STRUCTURE
# ─────────────────────────────────────────────
#
# Format (4 semicolon-delimited fields):
#
#   <sha256_hex> ; <nonce_b64> ; <tag_b64> ; <ciphertext_b64>
#
#   sha256_hex   : 64-char hex of original plaintext (integrity, plaintext AAD)
#   nonce_b64    : base64url of 16-byte AES-GCM nonce (plaintext, not secret)
#   tag_b64      : base64url of 16-byte GCM authentication tag
#   ciphertext   : AES-256-GCM encrypted payload containing:
#                    [4 bytes big-endian] message_length_bytes
#                    [4 bytes big-endian] original_bit_length
#                    [remaining bytes]   Elias Gamma compressed gap bytes
#
# message_length and elias_bit_length are INSIDE the ciphertext —
# an attacker cannot determine message length from auxiliary information.
# ─────────────────────────────────────────────

def build_auxiliary_information(message_hash, nonce, tag, ciphertext):
    """
    Construct the complete auxiliary information string for transmission.

    All byte fields are base64url-encoded for safe transmission over
    any text channel.  The SHA-256 hash is transmitted in plaintext as
    associated data — it is authenticated via the GCM tag so an attacker
    cannot substitute a different hash without failing verification.
    """
    return ";".join([
        message_hash,
        base64.urlsafe_b64encode(nonce).decode(),
        base64.urlsafe_b64encode(tag).decode(),
        base64.urlsafe_b64encode(ciphertext).decode()
    ])


def parse_auxiliary_information(auxiliary_string):
    """
    Parse auxiliary information string received from sender.

    Returns:
        message_hash : 64-char SHA-256 hex digest
        nonce        : 16 raw bytes
        tag          : 16 raw bytes (GCM auth tag)
        ciphertext   : raw bytes (encrypted payload)
    """
    parts = auxiliary_string.split(';', 3)
    if len(parts) != 4:
        raise ValueError(
            f"Malformed auxiliary information. Expected 4 fields, got {len(parts)}."
        )
    message_hash = parts[0]
    nonce        = base64.urlsafe_b64decode(parts[1])
    tag          = base64.urlsafe_b64decode(parts[2])
    ciphertext   = base64.urlsafe_b64decode(parts[3])

    if len(message_hash) != 64:
        raise ValueError(
            f"SHA-256 hash should be 64 hex characters, got {len(message_hash)}."
        )
    if len(nonce) != 16:
        raise ValueError(f"AES nonce should be 16 bytes, got {len(nonce)}.")
    if len(tag) != 16:
        raise ValueError(f"GCM tag should be 16 bytes, got {len(tag)}.")

    return message_hash, nonce, tag, ciphertext


def find_gaps(message_bits, encryption_key, bitstream, N, K):
    gaps = []
    position_scanned = []
    total_scanned = 0

    feasible, stats = check_feasibility(bitstream, message_bits, encryption_key)
    print("statistics for feasibility")
    print(stats)

    # if not feasible:
    #     raise ValueError(
    #         f"Image is not feasible for embedding the message. "
    #         f"Matching probability p = {stats['p']:.4f} is too low. "
    #         f"Expected gap = {stats['expected_gap']:.2f} bits, "
    #         f"which exceeds the image capacity."
    #     )

    for i in range(K):
        target_bit = message_bits[i]
        start = phi(encryption_key, i, N)
        gap = 0
        for offset in range(N):
            pos = (start + offset) % N
            total_scanned += 1
            if bitstream[pos] == target_bit:
                break
            gap += 1
        else:
            raise ValueError(
                f"Could not find a matching bit for message bit {i} "
                f"after scanning all N={N} positions. "
                f"Check that the image is not a flat (all-zero or all-one) image."
            )

        gaps.append(gap)
        position_scanned.append(gap+1)
    print("average: ", sum(position_scanned)/len(position_scanned))
    return gaps, total_scanned, position_scanned

# ─────────────────────────────────────────────
# ALGORITHM 2 — NBSM ENCODER  E: (M, B, K) → L
# ─────────────────────────────────────────────

def nbsm_encode(image_path, message, encryption_key, verbose=False):
    """
    NBSM Encoder — implements E: (M, B, K) → L from Definition 1.

    For each message bit mᵢ:
        1. Call Φ(K, i) to get an independent key-derived start position
        2. Scan forward from that position until a bit matching mᵢ is found
        3. Record Gᵢ = number of positions scanned before the match

    This produces gap list L = (G₁, G₂, ..., GK).
    Each Gᵢ ~ Geometric(0.5) independently — satisfying Theorems 1, 2, 3.

    The gap list is then:
        - Compressed with Elias Gamma  (Theorem 3: ≈ 2.77K bits total)
        - Packed into bytes with a length header
        - Encrypted with AES-256-GCM  (AEAD — hides length, detects tampering)
        - Combined with SHA-256 hash and nonce → auxiliary information

    Args:
        image_path     : path to cover image (not transmitted)
        message        : plaintext secret message string
        encryption_key : shared secret key K (Definition 1)
        verbose        : print step-by-step output

    Returns:
        auxiliary_information : string ready for transmission
        gap_list              : raw gap list (for experiments)
        stats                 : encoding statistics dict
    """

    # ── Step 1: Generate cover image bit stream B ─────────────────────────
    bitstream = image_to_bitstream(image_path)
    N = len(bitstream)

    # ── Step 2: Convert message M to bit string ───────────────────────────
    message_bits = message_to_bits(message)
    K = len(message_bits)
    message_length_bytes = len(message.encode('utf-8'))

    if K > N // 2:
        raise ValueError(
            f"Message too large for this image. "
            f"Message requires {K} bits but image capacity bound is "
            f"N/2 = {N // 2} bits (Theorem 2). Use a larger image or a shorter message."
        )

    if verbose:
        print(f"[ENCODE] Image bit stream  N = {N:,} bits")
        print(f"[ENCODE] Message length    K = {K} bits ({message_length_bytes} bytes)")
        print(f"[ENCODE] Capacity bound  N/2 = {N // 2:,} bits")
        print(f"[ENCODE] Message bits      : {message_bits}")

    # ── Step 3: Compute SHA-256 hash of plaintext message ────────────────
    message_hash = compute_message_hash(message)

    if verbose:
        print(f"[ENCODE] SHA-256 hash      : {message_hash}")

    # ── Step 4: Build gap list L using Φ(K, i) per message bit ───────────
    gaps, total_scanned, position_scanned = find_gaps(message_bits=message_bits, encryption_key=encryption_key, bitstream=bitstream, N=N, K=K)

    if verbose:
        print(f"[ENCODE] Bits matched      : {K}/{K}")
        print(f"[ENCODE] Total scanned     : {total_scanned:,}")
        print(f"[ENCODE] First 10 gaps     : {gaps[:10]}")
        print(f"[ENCODE] Mean gap          : {np.mean(gaps):.4f}  (Theorem 1: E[G] = 2)")
        print(f"[ENCODE] Var  gap          : {np.var(gaps):.4f}   (Theorem 1: Var[G] = 2)")

    # ── Step 5: Compress gap list with Elias Gamma  (Theorem 3) ──────────
    compressed_bits = compress_gaps(gaps)
    elias_bits      = len(compressed_bits)

    if verbose:
        print(f"[ENCODE] Elias key size    : {elias_bits} bits  "
              f"({elias_bits / K:.4f} bits/message-bit, Theorem 3 predicts ≈ 2.77)")
        print(f"[ENCODE] Entropy lower bnd : {2 * K} bits  (H(G) = 2 bits, Theorem 3)")
        print(f"[ENCODE] Overhead above H  : {elias_bits - 2 * K} bits  "
              f"(within 0.77K = {0.77 * K:.1f} bits of optimum)")

    # ── Step 6: Pack sensitive fields into one payload ────────────────────
    #
    #   Layout:  [4 bytes: message_length_bytes]
    #            [4 bytes: original_bit_length ]
    #            [N bytes: gap bytes           ]
    #
    #   Encrypting message_length_bytes and original_bit_length alongside
    #   the gap data closes the length side-channel — a passive attacker
    #   cannot determine how long the hidden message is.

    gap_bytes, original_bit_length = _bits_to_bytes(compressed_bits)

    length_header = _struct.pack('>II',
        message_length_bytes,
        original_bit_length
    )
    payload = length_header + gap_bytes

    # ── Step 7: Encrypt payload with AES-256-GCM ─────────────────────────
    #
    #   associated_data = SHA-256 hash of the message (transmitted in plaintext).
    #   Binding the hash to the ciphertext via GCM AAD means an attacker cannot
    #   substitute a different hash without failing the authentication tag check.

    associated_data = message_hash.encode('utf-8')
    nonce, tag, ciphertext  = aes_encrypt(payload, encryption_key, associated_data)

    if verbose:
        print(f"[ENCODE] AES-GCM nonce     : {nonce.hex()}")
        print(f"[ENCODE] GCM tag           : {tag.hex()}")
        print(f"[ENCODE] Ciphertext length : {len(ciphertext)} bytes")

    # ── Step 8: Build auxiliary information ──────────────────────────────
    auxiliary_information = build_auxiliary_information(
        message_hash, nonce, tag, ciphertext
    )

    # ── Statistics ───────────────────────────────────────────────────────
    stats = {
        'N_cover_bits'            : N,
        'K_message_bits'          : K,
        'message_length_bytes'    : message_length_bytes,
        'total_bits_scanned'      : total_scanned,
        'scan_efficiency'         : total_scanned / N,
        'mean_gap'                : float(np.mean(gaps)),
        'var_gap'                 : float(np.var(gaps)),
        'elias_key_bits'          : elias_bits,
        'bits_per_message_bit'    : elias_bits / K,
        'entropy_lower_bound_bits': 2 * K,
        'overhead_above_entropy'  : elias_bits - 2 * K,
        'aes_ciphertext_bytes'    : len(ciphertext),
        'auxiliary_total_chars'   : len(auxiliary_information),
        'sha256_hash'             : message_hash,
    }

    return auxiliary_information, gaps, stats


# ─────────────────────────────────────────────
# ALGORITHM 3 — NBSM DECODER  D: (L, B, K) → M
# ─────────────────────────────────────────────

def nbsm_decode(image_path, auxiliary_information, encryption_key, verbose=False):
    """
    NBSM Decoder — implements D: (L, B, K) → M from Definition 1.

    Reconstructs the message by replaying Φ(K, i) for each bit index i,
    then advancing Gᵢ steps from that start position to reach the matched
    position in B.  The bit value at that position is message bit mᵢ.

    This is the exact inverse of nbsm_encode — using the same Φ(K, i)
    function guarantees D(E(M, B, K), B, K) = M (correctness property).

    Args:
        image_path            : same cover image used during encoding
        auxiliary_information : complete auxiliary string from sender
        encryption_key        : same shared key K used during encoding
        verbose               : print step-by-step output

    Returns:
        recovered_message : verified recovered plaintext string

    Raises:
        ValueError : if AES-GCM authentication fails or SHA-256 check fails
    """

    # ── Step 1: Parse auxiliary information ──────────────────────────────
    message_hash, nonce, tag, ciphertext = \
        parse_auxiliary_information(auxiliary_information)

    if verbose:
        print(f"[DECODE] Expected hash     : {message_hash}")
        print(f"[DECODE] AES-GCM nonce     : {nonce.hex()}")
        print(f"[DECODE] Ciphertext length : {len(ciphertext)} bytes")

    # ── Step 2: Decrypt and verify with AES-256-GCM ──────────────────────
    #
    #   associated_data must match what the encoder used — SHA-256 hash.
    #   Wrong key or tampered data → GCM tag mismatch → ValueError here,
    #   before any gap processing occurs.  This is an early, hard rejection.

    associated_data = message_hash.encode('utf-8')
    payload         = aes_decrypt(nonce, tag, ciphertext, encryption_key, associated_data)

    # ── Step 3: Unpack length header and gap bytes from payload ──────────
    #
    #   First 8 bytes are the length header packed by the encoder:
    #     [0:4] message_length_bytes  (big-endian uint32)
    #     [4:8] original_bit_length   (big-endian uint32)
    #   Remaining bytes are the Elias Gamma compressed gap data.

    if len(payload) < 8:
        raise ValueError(
            f"Decrypted payload too short ({len(payload)} bytes). "
            f"Expected at least 8 bytes for the length header."
        )

    message_length_bytes, original_bit_length = _struct.unpack('>II', payload[:8])
    gap_bytes = payload[8:]
    K         = message_length_bytes * 8

    if verbose:
        print(f"[DECODE] Message length    : {message_length_bytes} bytes ({K} bits)")
        print(f"[DECODE] Elias bit length  : {original_bit_length} bits")

    # ── Step 4: Unpack bytes to bit string ───────────────────────────────
    compressed_bits = _bytes_to_bits(gap_bytes, original_bit_length)

    if verbose:
        print(f"[DECODE] Decrypted bits    : {len(compressed_bits)} bits")

    # ── Step 5: Decompress gap list ───────────────────────────────────────
    gaps = decompress_gaps(compressed_bits, K)

    if verbose:
        print(f"[DECODE] Gaps decoded      : {len(gaps)}")
        print(f"[DECODE] First 10 gaps     : {gaps[:10]}")
        print(f"[DECODE] Mean gap          : {np.mean(gaps):.4f}  (Theorem 1: E[G] = 2)")

    # ── Step 6: Generate cover image bit stream B ─────────────────────────
    bitstream = image_to_bitstream(image_path)
    N         = len(bitstream)

    if verbose:
        print(f"[DECODE] Image bit stream  N = {N:,} bits")

    # ── Step 7: Recover message bits using Φ(K, i) + gap list ────────────
    #
    #   For each bit index i:
    #     start       = Φ(encryption_key, i, N)   — same start as encoder
    #     matched_pos = (start + Gᵢ) % N           — advance Gᵢ steps
    #     mᵢ          = B[matched_pos]              — read bit at position

    recovered_bits = ''

    for i, gap in enumerate(gaps):
        start       = phi(encryption_key, i, N)
        matched_pos = (start + gap) % N
        recovered_bits += bitstream[matched_pos]

    if verbose:
        print(f"[DECODE] Recovered bits    : {len(recovered_bits)} bits")
        print(f"[DECODE] Recovered bits    : {recovered_bits}")

    # ── Step 8: Reconstruct message string ───────────────────────────────
    recovered_message = bits_to_message(recovered_bits)

    if verbose:
        print(f"[DECODE] Recovered message : {recovered_message}")

    # ── Step 9: SHA-256 Integrity Verification ────────────────────────────
    #
    #   Secondary check — the GCM tag already guarantees integrity.
    #   This provides an additional application-level confirmation that the
    #   recovered plaintext matches the original message exactly.

    hash_valid = verify_message_hash(recovered_message, message_hash)

    if verbose:
        computed_hash = compute_message_hash(recovered_message)
        print(f"[DECODE] Computed hash     : {computed_hash}")
        print(f"[DECODE] Expected hash     : {message_hash}")
        print(f"[DECODE] Hash match        : {'✓ VERIFIED' if hash_valid else '✗ FAILED'}")

    if not hash_valid:
        raise ValueError(
            "SHA-256 integrity verification FAILED. "
            "The recovered message does not match the transmitted hash. "
            "Possible causes: wrong encryption key, corrupted transmission, "
            "different cover image used, or message tampering."
        )

    return recovered_message


# ─────────────────────────────────────────────
# DEMO AND VALIDATION
# ─────────────────────────────────────────────



if __name__ == "__main__":

    IMAGE_PATH     = "../assets/black-dog-gray-srgb.png"
    SECRET_MESSAGE = "Hello, NBSM World!"
    SHARED_KEY     = "Hemraj@321"

    print("=" * 60)
    print("NBSM")
    print("=" * 60)

    # ── ENCODING ──────────────────────────────────────────────────────────
    print("\n[PHASE 1] ENCODING")
    print("-" * 60)

    

    auxiliary_info, gap_list, stats = nbsm_encode(
        IMAGE_PATH,
        SECRET_MESSAGE,
        SHARED_KEY,
        verbose=True
    )

    print(f"\n--- Encoding Statistics ---")

    print("Auxilary Info: ", auxiliary_info)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.6f}")
        else:
            print(f"  {k:35s}: {v}")

    # ── DECODING ──────────────────────────────────────────────────────────
    print("\n[PHASE 2] DECODING")
    print("-" * 60)

    recovered = nbsm_decode(
        IMAGE_PATH,
        auxiliary_info,
        SHARED_KEY,
        verbose=True
    )

    # ── FINAL VERIFICATION ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"  Original message  : {SECRET_MESSAGE}")
    print(f"  Recovered message : {recovered}")
    print(f"  Exact match       : {SECRET_MESSAGE == recovered}")

    # ── WRONG KEY TEST ────────────────────────────────────────────────────
    print("\n[PHASE 3] WRONG KEY ATTACK TEST")
    print("-" * 60)
    print("  Attempting decryption with wrong key...")
    try:
        nbsm_decode(
            IMAGE_PATH,
            auxiliary_info,
            "wrong_key_attempt",
            verbose=False
        )
        print("  Result: SECURITY FAILURE — wrong key should not succeed")
    except ValueError as e:
        print(f"  Result: ✓ Correctly rejected — {e}")
