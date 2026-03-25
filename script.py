from experiments.experiment1v4 import experiment

# ── Configuration ──────────────────────────────────────────────────
DATABASES = [["./datasets/BOSSbase", 1000, "BOSSbase, grayscale"],["./datasets/UCID1338", 100, "UCID1338 RGB"]]

MESSAGES     =  [
    ("Natural Bit Sequence Matching experiment message. " * 20)[:1024],          # 1KB
    ("Natural Bit Sequence Matching experiment message. " * 215)[:10240],        # 10KB
    ("Natural Bit Sequence Matching experiment message. " * 2200)[:102400],      # 100KB
    ("Natural Bit Sequence Matching experiment message. " * 22000)[:1048576],    # 1MB
] # exactly 10240 bytes = 81920 bits

experiment(DATABASES, MESSAGES, output_path="./experiments/results")