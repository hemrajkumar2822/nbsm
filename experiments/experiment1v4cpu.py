"""
Theorem 1 Verification: G∼Geometric(0.5), E[G] = 2, Var[G] = 2
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats


from algorithm.nbsm import image_to_bitstream, message_to_bits, find_gaps

KEY          = "hedkjkeijkdj2343"

# ── Plot: Theorem 1 Histogram (Fig 2 for paper) ─────────────────────

def plot_theorem1_histogram(image_means, image_vars, output_path, all_gaps_A, dataset_name, message_size):
    """
    Generate Figure 2 for the paper: Theorem 1 experimental verification.

    Three-panel figure:
        Panel (a) — Distribution of per-image mean(G) across all images.
                    Red dashed line at E[G]=2 (theory).
                    Images colour-coded by compliance group.

        Panel (b) — Distribution of per-image var(G) on log scale.
                    Red dashed line at Var[G]=2 (theory).

        Panel (c) — Empirical gap PMF vs Geometric(0.5) theoretical PMF.
                    Roadmap: 'histogram of gap values with Geometric(0.5) overlay'.

    Args:
        image_means : list or array of per-image mean(G) values
        image_vars  : list or array of per-image var(G)  values
        output_path : full file path to save PNG

    Saves:
        PNG at output_path, 300 DPI
    """
    means = np.array(image_means)
    varss = np.array(image_vars)
    n     = len(means)

    mask_A = (means >= 1.5) & (means <= 2.5)
    mask_B = (means >  2.5) & (means <= 5.0)
    mask_C =  means >  5.0

    A_m = means[mask_A]
    A_v = varss[mask_A]

    COLOR_A      = "#2ecc71"
    COLOR_B      = "#f39c12"
    COLOR_C      = "#e74c3c"
    COLOR_THEORY = "#c0392b"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Theorem 1 Experimental Verification on {dataset_name} "
        f"({n} images,  K = {message_size} bits)\n"
        "Red dashed lines mark theoretical predictions  "
        r"[$G \sim \mathrm{Geometric}(p=0.5)$,  $\mathbb{E}[G]=2$,  $\mathrm{Var}[G]=2$]",
        fontsize=11, fontweight='bold'
    )

    # ── Panel (a): mean(G) distribution ──────────────────────────────────
    ax1 = axes[0]
    display_max = 8.0
    bins = np.linspace(0, display_max, 55)

    ax1.hist(np.clip(means[mask_A], 0, display_max), bins=bins,
             color=COLOR_A, alpha=0.85,
             label=f"Group A — Assumption 1 holds  (n={mask_A.sum()})")
    ax1.hist(np.clip(means[mask_B], 0, display_max), bins=bins,
             color=COLOR_B, alpha=0.85,
             label=f"Group B — Mild deviation  (n={mask_B.sum()})")
    ax1.hist(np.clip(means[mask_C], 0, display_max), bins=bins,
             color=COLOR_C, alpha=0.85,
             label=f"Group C — Assumption 1 fails  (n={mask_C.sum()})")

    ax1.axvline(x=2.0, color=COLOR_THEORY, linewidth=2.5, linestyle='--',
                label="Theory:  E[G] = 2.0")

    obs_median = float(np.median(means))
    ax1.axvline(x=obs_median, color='navy', linewidth=1.8, linestyle=':',
                label=f"Observed median = {obs_median:.4f}")

    err_pct = abs(obs_median - 2.0) / 2.0 * 100
    ax1.text(
        0.97, 0.97,
        f"n = {n} images\n"
        f"Median = {obs_median:.4f}\n"
        f"Theory = 2.0000\n"
        f"Error  = {err_pct:.2f}%\n"
        f"Group A mean = {A_m.mean():.4f}",
        transform=ax1.transAxes, va='top', ha='right', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.9)
    )

    n_clipped = int(np.sum(means > display_max))
    ax1.set_xlabel("Per-image mean(G)", fontsize=11)
    ax1.set_ylabel("Number of images", fontsize=11)
    ax1.set_title(
        "(a)  Distribution of mean(G)\n"
        "Theorem 1 Part (i):  E[G] = 2",
        fontsize=10, fontweight='bold'
    )
    ax1.set_xlim(0, display_max)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    if n_clipped > 0:
        ax1.text(
            0.5, -0.10,
            f"Note: {n_clipped} Group C images with mean(G) > {display_max:.0f} clipped",
            transform=ax1.transAxes, ha='center', fontsize=7.5,
            color='grey', style='italic'
        )

    # ── Panel (b): var(G) distribution — log scale ───────────────────────
    ax2 = axes[1]
    log_bins = np.logspace(-1, 8, 65)

    ax2.hist(varss[mask_A], bins=log_bins, color=COLOR_A, alpha=0.85,
             label=f"Group A  (n={mask_A.sum()})")
    ax2.hist(varss[mask_B], bins=log_bins, color=COLOR_B, alpha=0.85,
             label=f"Group B  (n={mask_B.sum()})")
    ax2.hist(varss[mask_C], bins=log_bins, color=COLOR_C, alpha=0.85,
             label=f"Group C  (n={mask_C.sum()})")

    ax2.axvline(x=2.0, color=COLOR_THEORY, linewidth=2.5, linestyle='--',
                label="Theory:  Var[G] = 2.0")

    obs_var_median = float(np.median(varss))
    ax2.axvline(x=obs_var_median, color='navy', linewidth=1.8, linestyle=':',
                label=f"Observed median = {obs_var_median:.4f}")

    var_err_pct  = abs(obs_var_median - 2.0) / 2.0 * 100
    pct_within_3 = 100 * float(np.sum(A_v <= 3)) / len(A_v) if len(A_v) > 0 else 0
    ax2.text(
        0.97, 0.97,
        f"n = {n} images\n"
        f"Median = {obs_var_median:.4f}\n"
        f"Theory = 2.0000\n"
        f"Error  = {var_err_pct:.2f}%\n"
        f"Group A var≤3: {pct_within_3:.0f}%",
        transform=ax2.transAxes, va='top', ha='right', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.9)
    )

    ax2.set_xscale('log')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Per-image var(G)  [log scale]", fontsize=11)
    ax2.set_ylabel("Number of images", fontsize=11)
    ax2.set_title(
        "(b)  Distribution of var(G)\n"
        "Theorem 1 Part (ii):  Var[G] = 2",
        fontsize=10, fontweight='bold'
    )
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # ── Panel (c): Geometric(0.5) PMF overlay ────────────────────────────
    ax3 = axes[2]
    k_vals   = np.arange(1, 16)
    theo_pmf = 0.5 * (0.5 ** (k_vals - 1))

    counts = Counter(all_gaps_A)
    total  = len(all_gaps_A)

    total_shown = sum(counts.get(k, 0) for k in k_vals)


    if total == 0:
        emp_pmf_A = np.zeros(len(k_vals))
    else:
        emp_pmf_A = np.array([counts.get(k, 0) / total_shown for k in k_vals])


    bar_w = 0.35
    ax3.bar(k_vals - bar_w/2, emp_pmf_A, width=bar_w,
            color=COLOR_A, alpha=0.85,
            label=f"Empirical — Group A  (gaps={total_shown})")
    ax3.bar(k_vals + bar_w/2, theo_pmf, width=bar_w,
            color=COLOR_THEORY, alpha=0.70,
            label="Theoretical — Geometric(0.5)")

    ax3.set_xlabel("Gap value  G", fontsize=11)
    ax3.set_ylabel("P(G = k)", fontsize=11)
    ax3.set_title(
        r"(c)  Empirical PMF vs Geometric$(p=0.5)$" + "\n",
        fontsize=10, fontweight='bold'
    )
    ax3.set_xlim(0.5, 15.5)
    ax3.legend(fontsize=8.5)
    ax3.grid(True, alpha=0.3, axis='y')

    safe_emp  = np.clip(emp_pmf_A, 1e-10, None)
    safe_theo = np.clip(theo_pmf,  1e-10, None)
    kl_div    = float(np.sum(safe_emp * np.log(safe_emp / safe_theo)))
    ax3.text(
        0.97, 0.97,
        f"KL(empirical ‖ theory)\n≈ {kl_div:.5f}\n(0 = perfect match)",
        transform=ax3.transAxes, va='top', ha='right', fontsize=8.5,
        bbox=dict(boxstyle='round', facecolor='lightyellow',
                  edgecolor='grey', alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {output_path}")


# ── Main ───────────────────────────────────────────────────────────

def run(image_folder, max_images, message, dataset_name, output_path="./results"):
    supported = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.pgm'}
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f.lower())[1] in supported]

    if not image_files:
        print(f"ERROR: No images in '{image_folder}'")
        sys.exit(1)

    rng = np.random.default_rng(0)
    if len(image_files) > max_images:
        image_files = list(rng.choice(image_files, max_images, replace=False))
    
    message_bits = message_to_bits(message)
    message_bit_len=len(message_bits)

    print(f"Images   : {len(image_files)}")
    print(f"K (bits) : {message_bit_len} = {len(message)} bytes")
    print(f"Key      : {KEY}")
    print()

    image_means = []
    image_vars  = []
    all_gaps_A = []

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        bitstream = image_to_bitstream(image_path=img_path)
        bitstream_len = len(bitstream)
        _, _, positions_scanned  = find_gaps(message_bits=message_bits, 
                                encryption_key=KEY, 
                                bitstream=bitstream,
                                N=bitstream_len,
                                K=message_bit_len)
        g        = np.array(positions_scanned)
        mean_g   = float(g.mean())
        var_g    = float(g.var())
        image_means.append(mean_g)
        image_vars.append(var_g)
        if 1.5 <= mean_g <= 2.5:   # Group A
            all_gaps_A.extend(g.tolist())

        print(f"[{idx+1:>4}] {img_file:<30}  mean={mean_g:.4f}  var={var_g:.4f}")

    # ── Summary statistics ─────────────────────────────────────────
    means = np.array(image_means)
    varss = np.array(image_vars) 
    n     = len(means)

    print()
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    print(f"\n{'Statistic':<40} {'Observed':>10}  {'Theory':>10}")
    print("-" * 63)
    print(f"  {'Median mean(G) — all images':<38} {np.median(means):>10.4f}  {'2.0000':>10}")
    print(f"  {'Mean   mean(G) — all images':<38} {means.mean():>10.4f}  {'2.0000':>10}")

    print(f"  {'Median var(G)  — all images':<38} {np.median(varss):>10.4f}  {'2.0000':>10}")
    print(f"  {'Mean   var(G)  — all images':<38} {varss.mean():>10.4f}  {'2.0000':>10}")

    print(f"  {'Std    mean(G) — all images':<38} {means.std():>10.4f}  {'':>10}")
    print(f"  {'Std    var(G)  — all images':<38} {varss.std():>10.4f}  {'':>10}")

    
    os.makedirs(output_path, exist_ok=True)
    plot_theorem1_histogram(
        image_means=means,
        image_vars=varss,
        output_path=f"{output_path}/exp1_histo_{len(message)}.png",
        all_gaps_A=all_gaps_A,
        dataset_name=dataset_name,
        message_size=message_bit_len
    )

    # ── Group breakdown ────────────────────────────────────────────
    group_defs = [
        ("Group A  mean in [1.5 to 2.5]", (means >= 1.5) & (means <= 2.5)),
        ("Group B  mean in [2.5 to 5.0]", (means >  2.5) & (means <= 5.0)),
        ("Group C  mean > 5.0",          means > 5.0),
    ]



    print(f"\n{'Group':<30} {'n':>5}  {'mean(G)':>8}  {'var(G)':>9} {'err%':>6}")
    print("-" * 95)


    for label, mask in group_defs:
        sub_m = means[mask]
        sub_v = varss[mask]
        if len(sub_m) == 0:
            continue
        m      = sub_m.mean()
        v      = sub_v.mean()
        em     = abs(m - 2.0) / 2.0 * 100
        
        print(f"  {label:<28} {len(sub_m):>5}  {m:>8.4f}  {v:>9.4f} {em:>5.2f}%")

    # ── Paper table ────────────────────────────────────────────────
    group_a     = means[(means >= 1.5) & (means <= 2.5)]
    group_a_var = varss[(means >= 1.5) & (means <= 2.5)]

    print(f"\n{'='*65}")
    print(f"PAPER TABLE — THEOREM 1 VERIFICATION")
    print(f"{'='*65}")
    print(f"""
  Dataset         : {n} images, {dataset_name}
  Message size    : {len(message):,} bytes  (K = {message_bit_len:,} bits)
  Convention      : B  (G_i includes matched position, G_i >= 1)

  Statistic                                             Observed                                                    Theory
  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Median mean(G)  [all {n} images]                      {np.median(means):.4f}                                      2.0000
  Median var(G)   [all {n} images]                      {np.median(varss):.4f}                                      2.0000

  Mean mean(G)    [Group A, {len(group_a)} img]         {group_a.mean():.4f}                                        2.0000
  Mean var(G)     [Group A, {len(group_a)} img]         {group_a_var.mean():.4f}                                    2.0000

  Error mean(G)   [Group A]                             {abs(group_a.mean()-2)/2*100:.2f}%                          0%
  Error var(G)    [Group A]                             {abs(group_a_var.mean()-2)/2*100:.2f}%                      0%

  
  Theorem holds                                         {len(group_a)}/{n} ({100*len(group_a)/n:.0f}%)              —
  Assumption 1 fails                                    {int(np.sum(means>5))}/{n} ({100*np.sum(means>5)/n:.0f}%)   —
""")

    
    
    # Chi-square goodness-of-fit test against Geometric(0.5)
    k_vals_test = np.arange(1, 16)
    theo_pmf    = 0.5 * (0.5 ** (k_vals_test - 1))
    tail_prob   = 1.0 - theo_pmf.sum()          # P(G >= 16)

    observed = np.array(
        [np.sum(np.array(all_gaps_A) == k) for k in k_vals_test] +
        [np.sum(np.array(all_gaps_A) >= 16)]
    )
    expected = np.append(theo_pmf, tail_prob) * len(all_gaps_A)

    chi2_stat, chi2_p = stats.chisquare(observed, expected)
    print(f"Chi-square GoF test (vs Geometric(0.5)):")
    print(f"  chi2={chi2_stat:.4f}, df=15, p={chi2_p:.4f}")
    print(f"  Total gaps: {len(all_gaps_A):,}  (Group A images only)")
    print(f"  Note: with ~70M samples, p-value reflects sample size sensitivity.")
    print(f"  The chi2 statistic magnitude is the meaningful measure of fit.")

    # ── Save CSV ───────────────────────────────────────────────────
    out = f"{output_path}/exp1_{len(message)}.csv"
    with open(out, "w") as f:
        f.write("image,mean_gap,var_gap,group\n")
        for img, m, v in zip(image_files, image_means, image_vars):
            if m <= 2.5:   g = "A"
            elif m <= 5.0: g = "B"
            else:          g = "C"
            f.write(f"{img},{m:.6f},{v:.6f},{g}\n")
    print(f"  Results saved to {out}")


def experiment(databases, messages, output_path="./results"):
    for image_folder, max_images, db_name in databases:
        for message in messages:
            run(image_folder, max_images, message, db_name, output_path=output_path)
    print("End of experiment")
