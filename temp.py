from algorithm.nbsm import image_to_bitstream, message_to_bits


def check_feasibility2(bitstream, msg_bits):
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



def calculate_p_value(bitstream, msg_bits):
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
    p = check_feasibility(bitstream, msg_bits)
    p2 = check_feasibility2(bitstream, msg_bits)
    print(f"Estimated matching probability p: {p:.4f}")
    print(f"Estimated matching probability p (analytical): {p2:.4f}")