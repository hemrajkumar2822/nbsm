from algorithm.nbsm import image_to_bitstream, message_to_bits
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


# def check_feasibility(bitstream, msg_bits, safety_factor=1.2, min_p=0.01):
#     N = len(bitstream)
#     K = len(msg_bits)
#     # ── Basic validation ─────────────────────────────────────────
#     if N == 0 or K == 0:
#         return False, {"reason": "Empty bitstream or message"}

#     # ── Count bits (fast, no loop over both) ─────────────────────
#     ones_B = bitstream.count('1')
#     ones_M = msg_bits.count('1')

#     # ── Probabilities ────────────────────────────────────────────
#     P_B1 = ones_B / N
#     P_M1 = ones_M / K

#     # Avoid recomputing zeros
#     P_B0 = 1.0 - P_B1
#     P_M0 = 1.0 - P_M1

#     # ── Matching probability ─────────────────────────────────────
#     p = P_B0 * P_M0 + P_B1 * P_M1

#     # ── Edge cases ───────────────────────────────────────────────
#     if p <= min_p:
#         return False, {
#             "p": p,
#             "reason": "Matching probability too low"
#         }

#     # ── Expected gap ─────────────────────────────────────────────
#     expected_gap = 1.0 / p

#     # ── Required scan length ─────────────────────────────────────
#     required_bits = K * expected_gap # * safety_factor

#     feasible = required_bits <= N

#     stats = {
#         "p": float(p),
#         "expected_gap": float(expected_gap),
#         "required_bits": float(required_bits),
#         "available_bits": int(N),
#         "message_bits": int(K),
#         "utilization_ratio": float(required_bits / N)
#     }

#     print("Feasibility check stats:")
#     # print(stats)

#     return feasible, stats


bitstream = image_to_bitstream("./assets/lion.jpg")
msg_bits = message_to_bits("Check whether the given image can feasibly embed the message using NBSM (gap-based matching). Uses empirical matching probability p to estimate expected gaps.")


# def check_feasibility(bitstream, msg_bits):
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

   
#     return p



def check_feasibility2(bitstream, msg_bits, safety_factor=1.2, min_p=0.01):
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

    return p



if __name__ == "__main__":
    # Example usage
    # p = check_feasibility(bitstream, msg_bits)
    p2 = check_feasibility2(bitstream, msg_bits)
    # print(f"Estimated matching probability p: {p:.4f}")
    print(f"Estimated matching probability p (analytical): {p2:.4f}")