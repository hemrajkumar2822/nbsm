from algorithm.nbsm import image_to_bitstream


def estimate_p_from_image(bitstream):
    """
    p = P(cover bit = 1) directly from raw bit counts.
    This is the image-only property that Theorem 1 depends on.
    Under Assumption 1, p should be 0.5.
    """
    bits  = np.array(list(bitstream), dtype=np.uint8)
    ones  = int(np.sum(bits == 1))  
    N     = len(bits)
    p1    = ones / N        # P(bit = 1)
    p0    = 1.0 - p1        # P(bit = 0)
    # Matching probability when target bit is equally likely 0 or 1:
    # p = P(match) = P(cover=0)*P(msg=0) + P(cover=1)*P(msg=1)
    # For a uniform message: P(msg=0) = P(msg=1) = 0.5
    # p = 0.5 * p0 + 0.5 * p1 = 0.5 always -- not useful!
    # 
    # Instead use p = P(cover bit = 1) as the geometric parameter.
    # The gap G_i ~ Geometric(p_match) where p_match depends on target bit:
    #   when target=1: p_match = p1 (prob cover bit is 1)
    #   when target=0: p_match = p0 (prob cover bit is 0)
    # Overall E[G] = 0.5 * 1/p1 + 0.5 * 1/p0  (average over target bits)
    # NOT 1/p_match directly.
    return p1, p0


