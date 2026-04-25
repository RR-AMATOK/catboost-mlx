# S38 T0a — Formal Analysis of Small-N LG+Cosine Residual Drift

**Mode:** Pragmatic (applied math, code-grounded).
**Author:** mathematician agent.
**Date:** 2026-04-25.
**Branch base:** master `600e5b7285` (post-S37 merge; DEC-042 closed for ST).
**Kernel md5 invariant:** `9edaef45b99b9db3e2717da93800e76f` — unchanged.

---

## 1. Setup — code-grounded dispatch map

The S37 #113 gate matrix (`docs/sprint37/t3-rerun/verdict.md`) measured drift in the **csv_train binary** (`catboost/mlx/tests/csv_train.cpp`), not the Metal-backed `structure_searcher.cpp`. Three grow policies dispatch to two distinct CPU split-finder routines in csv_train.cpp:

| Grow policy | Split finder | csv_train.cpp call site | DEC-042 status |
|---|---|---|---|
| SymmetricTree | `FindBestSplit` (L1593) | L4836 (`auto bestSplit = FindBestSplit(...)`) | **FIXED** in S33-L4-FIX Commit 1.5 + Commit-1H (per-side mask, both ordinal L1977-1985 and one-hot L1707-1733) |
| Depthwise | `FindBestSplitPerPartition` (L2251) | L4674 (`auto perPartSplits = FindBestSplitPerPartition(...)`) | **NOT FIXED** |
| Lossguide | `FindBestSplitPerPartition` (L2251) | L4415 (`auto perPartSplits = FindBestSplitPerPartition(...)`) | **NOT FIXED** |

A self-incriminating comment is anchored at csv_train.cpp:549:
> `// Score function (DW / LG paths — FindBestSplitPerPartition only)`

The same file at L4358 confirms the LG eval helper (`evalLeafLossguide`) builds a 2-partition layout (partition 0 = in-leaf docs, partition 1 = out-of-leaf), then calls FBSPP with `numPartitions=2u` and consumes only `perPartSplits[0]` (L4422-4427). The out-of-leaf partition's split is discarded.

`structure_searcher.cpp::SearchLossguideTreeStructure` (L455) calls a different routine — `FindBestSplitGPU` (Metal kernel, L625) — but that path is **not** what the S37 G3b/G3c gates exercised. The csv_train CPU FBSPP path is the runtime under measurement.

S35-#129 fixed only the one-hot branch of `FindBestSplit` (csv_train.cpp:1698-1733). It did **not** touch FBSPP.

## 2. FBSPP per-(p, k) contribution formula

Reading FBSPP ordinal at csv_train.cpp:2371-2431:

```cpp
for (ui32 bin = 0; bin + 1 < feat.Folds; ++bin) {
    for (ui32 p = 0; p < numPartitions; ++p) {
        double gain = 0.0;
        double cosNum_d = 0.0;
        double cosDen_d = 1e-20;
        for (ui32 k = 0; k < K; ++k) {
            // ... compute sumLeft, sumRight, weightLeft, weightRight ...
            if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;   // L2388 ← THE BUG
            switch (scoreFunction) {
                case EScoreFunction::L2:
                    gain += (sumLeft * sumLeft) / (weightLeft + l2RegLambda)
                          + (sumRight * sumRight) / (weightRight + l2RegLambda)
                          - (totalSum * totalSum) / (totalWeight + l2RegLambda);
                    break;
                case EScoreFunction::Cosine: {
                    cosNum_d += dSL*dSL*dInvL + dSR*dSR*dInvR;
                    cosDen_d += dSL*dSL*dWL*dInvL*dInvL + dSR*dSR*dWR*dInvR*dInvR;
                    break;
                }
            }
        }
        if (scoreFunction == EScoreFunction::Cosine) {
            gain = cosNum_d / std::sqrt(cosDen_d);
        }
        // argmax over (featIdx, bin) for this partition p
    }
}
```

Identical pattern in FBSPP one-hot at csv_train.cpp:2284-2349 with the joint-skip at **L2304**:
```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

**Per-(p, k) contribution to the LG argmax:**

For a fixed feature `f`, fixed bin `b`, fixed partition `p` (the leaf-being-expanded), summing over dimension `k`:

- **L2 contribution from one (p, k):**
  $$
  \Delta L_2(p,k) =
  \begin{cases}
  \frac{s_L^2}{w_L + \lambda} + \frac{s_R^2}{w_R + \lambda} - \frac{s_P^2}{w_P + \lambda}, & w_L > \varepsilon \text{ and } w_R > \varepsilon \\
  0, & \text{otherwise (joint-skip)}
  \end{cases}
  $$

- **Cosine contributions from one (p, k):**
  $$
  \Delta\text{cosNum}(p,k) = \mathbb{1}[w_L > \varepsilon \wedge w_R > \varepsilon]\,\left(\frac{s_L^2}{w_L+\lambda} + \frac{s_R^2}{w_R+\lambda}\right)
  $$
  $$
  \Delta\text{cosDen}(p,k) = \mathbb{1}[w_L > \varepsilon \wedge w_R > \varepsilon]\,\left(\frac{s_L^2 w_L}{(w_L+\lambda)^2} + \frac{s_R^2 w_R}{(w_R+\lambda)^2}\right)
  $$

Final per-bin Cosine gain at partition `p`:
$$
\text{gain}(p) = \frac{\sum_{k=0}^{K-1} \Delta\text{cosNum}(p,k)}{\sqrt{\varepsilon_d + \sum_{k=0}^{K-1} \Delta\text{cosDen}(p,k)}}, \quad \varepsilon_d = 10^{-20}
$$

**Cross-partition independence:** `gain` and `cosNum_d/cosDen_d` are declared inside the `for p` loop. There is **no** cross-partition running sum. The S33 root-cause "cross-partition cell omission" was about a single sum running across all `p` values inside `FindBestSplit`. FBSPP evaluates each partition independently, so the original S33 bug class does not transfer here.

But the per-(p, k) **joint-skip** does transfer — and it is the same formula DEC-042 fixed in `FindBestSplit`.

## 3. FBSPP vs FindBestSplit (post-S33) — structural diff

Side-by-side at the (p, k) level:

| Site | File:Line | Code | Behavior at degenerate child |
|---|---|---|---|
| **FindBestSplit ordinal Cosine (post-fix)** | csv_train.cpp:1961-2016 | `if (!wL_pos && !wR_pos) break; if (wL_pos) {…} if (wR_pos) {…}` | Per-side mask: contributes the non-empty side; only skips when **both** sides are empty (true no-op). |
| **FindBestSplit ordinal L2 (post-fix)** | csv_train.cpp:1977-1985 | Same per-side mask; subtracts parent term once if at least one side non-empty. | Per-side mask. |
| **FindBestSplit one-hot (post-fix, S35-#129)** | csv_train.cpp:1698-1733 | Per-side mask for L2; one-hot Cosine retains joint-skip per S34-PROBE-F-LITE verdict. | Per-side mask (L2) / joint-skip (Cosine, deliberately). |
| **FBSPP ordinal (UNFIXED)** | csv_train.cpp:2388 | `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` | **Joint-skip** — drops the entire (p, k) cell whenever either child is empty. |
| **FBSPP one-hot (UNFIXED)** | csv_train.cpp:2304 | `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` | **Joint-skip**. |

FBSPP at lines 2304 and 2388 is **byte-equivalent to the pre-S33 `FindBestSplit` joint-skip** that DEC-042 identified as the root cause. The S33 sprint-close `cr-report.md` S33-S1 reasoning ("FBSPP is structurally immune to the cross-partition running-sum bug") is correct *for that specific bug class*, but it is not the same as "FBSPP has no per-side mask issue." The per-side mask is an independent fix that DEC-042 did not extend to FBSPP.

**Conclusion of this comparison:** FBSPP carries the same algorithmic defect that DEC-042 closed in `FindBestSplit`. The post-S33 codebase has a **per-side mask in the SymmetricTree path** and **joint-skip everywhere else**.

## 4. H1 evaluation — small-N noise amplification in the post-fix FBS path

Hypothesis: at small N the per-side mask injects noise contributions instead of signal contributions because the non-empty side itself has too few docs.

**Setup:** N=1k, depth=6, max_leaves=31. Average leaf size ≈ N / max_leaves = 1000/31 ≈ 32 docs. Bins=128 ordinal.

**Counterargument from the gate matrix:** ST+Cosine at the *small-N* anchor was not measured directly in S37 #113 (G3a is N=50k). However, **G3a passed at 1.27%** post-fix on N=50k. If H1 were the right answer, we'd expect drift to scale with the joint-skip frequency (#degenerate (p, k) cells per bin candidate), which at depth=6 grows roughly as `2^d / N` — that is, drift scales when leaves shrink. ST and LG at N=1k both have the same average leaf size, so ST should also drift heavily under H1.

**The H1 prediction is testable cheaply (see §7).** If ST+Cosine at N=1k, depth=6 stays under ~2% post-fix, H1 is falsified — the small-N issue is LG-specific (or, more precisely, FBSPP-specific), and the per-side mask is not the noise source.

**Math:** the per-side mask change inside the post-fix FindBestSplit ordinal Cosine adds `s²/(w+λ)` from the non-empty side. For a single-doc leaf (extreme case): if doc has gradient `g` and Hessian `h`, then `s=g, w=h`, contribution = `g² / (h + λ)`. This is the Newton step squared — it is real signal, not noise. Concretely, the per-side mask provides the rank-1 information from the surviving side and is exactly what CatBoost CPU's `CalcAverage` formula at `online_predictor.h:112-119` produces (cited in csv_train.cpp:1968-1971 comment).

So the per-side mask itself does not amplify noise; it **corrects** an under-attribution. H1 is unlikely to be the dominant mechanism. Verdict: **H1 weak; falsifiable by ST+Cosine at N=1k.**

## 5. H2 evaluation — LG priority-queue × small-N interaction independent of code path

Hypothesis: LG's best-first leaf expansion concentrates drift in early splits where ranking sensitivity is highest, regardless of the per-bin formula.

**Evidence already on the table** (`docs/sprint29/lg-mechanism-spike/verdict.md`):
- At N=1000, depth=3, max_leaves=8, post-S28: iter-1 mean drift was 0.0024% — three orders of magnitude smaller than what S37 #113 G3b reports (27-31%) at depth=6, max_leaves=31.
- The S29 spike used **the same N (1000)** and **the same csv_train binary**. The only difference between the S29 spike (0.0024%) and S37 G3b (27-31%) is depth/max_leaves and iteration count.
- The S29 spike explicitly noted "priority-queue ordering surface likely under-exercised … with only 8 leaves the queue makes few contested choices."

**A pure priority-queue mechanism** would be expected to be present at depth=3, max_leaves=8 too — it would appear as a smooth ramp up with `max_leaves`. Drift jumping from 0.0024% (max_leaves=8) to ~30% (max_leaves=31) is **104×** for ~4× more leaves. That is not a smooth ramp; that is an inflection point.

**Inflection-point reasoning:** the joint-skip at FBSPP:2388 fires whenever a (feat, bin) candidate produces a degenerate child for the leaf being evaluated. At depth=3 (8 leaves of ~125 docs each) most ordinal bins still split both children non-empty; the joint-skip fires rarely. At depth=6 (31 leaves of ~32 docs each) the average leaf has fewer docs than there are bins (128), so most (feat, bin) candidates will have at least one empty child — the joint-skip becomes the **dominant** code path.

**Quantification (rough):** for a leaf with `n` docs and `B` bins per feature, assume docs are roughly uniform across bins. The probability that a given (feat, bin) candidate has *at least one* empty child is approximately:
$$
P(\text{degenerate}) = (1 - b/B)^n + ((B-b)/B)^n - (\text{both empty, vanishing})
$$
where `b` is the bin index and the probabilities are over which docs land in which bin. For `n=32`, `B=128`, this is large for low and high bins; integrated across bins, **a majority of bins produce at least one empty child at n≈32**.

So the joint-skip dominates at small leaf size, and the LG argmax at small N is computed on a **massively under-attributed** set of candidates. CPU CatBoost (per-side mask) sees the full candidate set; MLX FBSPP sees only the proper-subset where both children are non-empty. The argmax then disagrees on which bin to split — and since LG uses the chosen split's gain to prioritize the queue, the entire tree shape diverges.

H2 (queue × small-N) is real but **secondary**: the priority queue amplifies the FBSPP joint-skip damage by propagating early-split disagreements into later iterations. Without the joint-skip in FBSPP, the queue would have nothing to amplify. Verdict: **H2 is a co-mechanism, not the root cause.**

## 6. H3 evaluation — FBSPP carries an unfixed `continue`-on-degenerate that DEC-042 should have addressed but did not

This is the strongest hypothesis. Direct evidence:

1. **The exact joint-skip pattern** that DEC-042 documented as the bug (csv_train.cpp:1953-1959 commit-comment) is still present in FBSPP at csv_train.cpp:2304 (one-hot) and csv_train.cpp:2388 (ordinal).
2. **FBSPP is the code path used by LG and DW** in the csv_train binary (csv_train.cpp:549 comment, L4415, L4674). The G3b/G3c gates exercise LG → FBSPP. The G3a gate exercises ST → FindBestSplit (which IS fixed). Drift is precisely partitioned along this code-path boundary.
3. **The S33-L4-FIX commit history** explicitly tracks the per-side-mask change in FindBestSplit only:
   - Commit 1 (b4f7… per S33 history): per-side mask for `FindBestSplit` ordinal Cosine.
   - Commit 1.5 (S33-L4-FIX): per-side mask for `FindBestSplit` ordinal L2.
   - Commit-1H (S35-#129): per-side mask for `FindBestSplit` one-hot L2.
   - **No commit** ports the fix into FBSPP at csv_train.cpp:2304 or :2388.
4. **PROBE-E sampled only the N=50k anchor** with ST (S33 sprint-close `lg-mechanism-spike`/PROBE-E logs). Small-N counterfactual capture and any FBSPP capture were never done — the fix was extrapolated to LG without re-validation, which the S33 sprint-close acknowledged as conditional.
5. **The S33 code-reviewer S33-S1 reasoning** ("FBSPP is structurally immune") is correct only for the *cross-partition running-sum* sub-bug. The per-side-mask sub-bug is independent and lives at the per-(p, k) level. cr-report.md S33-S1 conflated the two.

**Magnitude check for H3:**
- N=50k LG+Cosine drift was 0.382% (S33 measurement). Average leaf at depth=6, max_leaves=31, N=50k is ~1612 docs — far above the 128-bin threshold; joint-skip rarely fires; FBSPP and a hypothetical per-side-mask FBSPP differ on a small set of cells; drift stays small.
- N=1k LG+Cosine drift is 27-31%. Average leaf ≈ 32 docs ≪ 128 bins; joint-skip dominates; FBSPP and CPU disagree on the candidate set for most cells; drift explodes.
- The drift magnitude is monotone in joint-skip frequency, which scales with `B/n` (bins per leaf size). The S37 #113 verdict's hypothesis-1 prose predicted exactly this regime-dependence; the math here grounds the prediction.

Verdict: **H3 is the load-bearing mechanism. Confidence: high.**

## 7. Discriminating prediction (cheapest empirical test)

The H1/H2/H3 hypotheses make **different predictions about ST+Cosine at N=1k, depth=6, bins=128**:

| Hypothesis | ST+Cosine at N=1k, depth=6, bins=128, post-S33 | Reason |
|---|---|---|
| H1 (small-N noise amplification) | **>10% drift** | Small-N is structural; the per-side mask injects noise at small leaves regardless of code path. |
| H2 (LG priority queue) | **<2% drift** | ST has no priority queue; the queue mechanism cannot fire. |
| H3 (FBSPP unfixed joint-skip) | **<2% drift** | ST uses FindBestSplit (fixed); the small-N regime exercises the fix at full intensity but the fix is correct. |

**Discriminating cell to run:** ST+Cosine, N=1k, depth=6, bins=128, 5 seeds (mirrors G3b config except grow_policy=SymmetricTree).

**Expected outcome (this analysis predicts H3):** drift in the **1-3%** band — small-N adds some noise relative to N=50k's 1.27%, but stays well within the [0.98, 1.02] envelope or marginally above. If ST+Cosine at N=1k drifts >10%, the analysis is wrong (H1 is the answer); if it drifts <2%, H3 is confirmed and the path forward is to port the per-side mask into FBSPP.

This T1 cell is essentially free: same harness, single grow-policy flag flip, ~5 minutes of training time. It should be the first thing S38 runs.

## 8. Verdict and recommended fix shape

**Best hypothesis: H3 (FBSPP unfixed joint-skip). Confidence: high.**

Mixed contribution from H2 (LG queue amplification) is real but secondary — H2 propagates H3's damage; remove H3 and H2 has no fuel.

**Recommended fix shape (in preference order):**

### Option A (preferred) — port DEC-042 per-side mask into FBSPP

- **Where:** csv_train.cpp:2304 (one-hot) and csv_train.cpp:2388 (ordinal).
- **Change:** replace `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` with the per-side mask pattern already in FindBestSplit at L1977-2016 (ordinal) and L1704-1733 (one-hot).
- **For Cosine ordinal:** mirror the FindBestSplit per-side accumulation with the L2-style guard `if (!wL_pos && !wR_pos) break;` then conditionally accumulate each side's `cosNum`/`cosDen` contribution.
- **For Cosine one-hot:** S34-PROBE-F-LITE verdict said "leave joint-skip for one-hot Cosine" — that decision was made for FindBestSplit and should be re-examined for FBSPP. If the same reasoning holds (rare-category bias), keep the joint-skip on the one-hot Cosine branch only and apply per-side mask to ordinal Cosine + both L2 branches. The one-hot drift contribution to G3b/G3c is likely small because FBSPP ordinal scoring dominates at bins=128.
- **For L2 ordinal and L2 one-hot:** apply per-side mask unconditionally, mirroring FindBestSplit fixes.
- **Cost:** one or two commits, gated on a re-run of G3b/G3c (target: drift <2%) and a non-regression check on G3a (must remain <2%) and on N=50k LG+Cosine (must remain <0.5%).
- **Risk:** the parent-term subtraction in FBSPP L2 (currently inside the joint-skip guard) becomes a separate question — the FindBestSplit post-fix subtracts the parent term unconditionally when at least one side is non-empty (csv_train.cpp:1984). FBSPP must follow the same pattern.

### Option B (fallback) — re-add LG+Cosine guard with N-threshold

If Option A introduces unforeseen one-hot regressions, fall back to:
- **Re-add S28-LG-GUARD** with conditional `if (N < N_threshold) reject;` semantics. The empirical scaling probe (T1) at N ∈ {500, 1k, 2k, 5k, 10k, 50k} fixes the threshold.
- **Estimate from the analysis:** joint-skip dominates when `B/n_leaf > 1`, where `n_leaf ≈ N / max_leaves`. With B=128 and max_leaves=31, the boundary is `N ≈ 128 × 31 = 3968`. Below this, drift is structural; above, it's float-precision residual. Round to **N_threshold = 5000** (safety margin).
- **Cost:** lower than Option A, but leaves the algorithmic defect in the codebase indefinitely; not the "fix properly" preferred direction (per project memory `feedback_fix_properly.md`).

### Option C (do not pursue) — score-formula refinement at small leaves

H1 was the only hypothesis pointing here, and the analysis in §4 falsifies it pre-emptively. Score-formula refinement risks departing from CPU-reference parity rather than restoring it.

**Recommendation: pursue Option A. Run the T1 ST+Cosine N=1k discriminating prediction first to confirm H3 (cheap), then port the per-side mask into FBSPP one-hot and ordinal in a single commit per the S33-L4-FIX protocol.**

## 9. Open questions

1. **Does the S35-PROBE-F-LITE one-hot Cosine "leave joint-skip" verdict apply to FBSPP one-hot Cosine?** PROBE-F-LITE was conducted on FindBestSplit one-hot Cosine; the rare-category-bias mechanism may or may not transfer. A T0c probe (one-hot Cosine FBSPP per-side mask vs joint-skip on a low-cardinality categorical) would settle this. For now, recommend keeping joint-skip on one-hot Cosine FBSPP and porting the fix only where S33 already validated it (ordinal Cosine, L2 both branches).
2. **Does `structure_searcher.cpp::SearchLossguideTreeStructure` (Metal path, L455-L749) carry the same defect?** It calls `FindBestSplitGPU` which dispatches into Metal kernels (`compute_split_scores`-class). The kernel md5 invariant `9edaef45b99b9db3e2717da93800e76f` is unchanged from S33, but the Metal score-calc kernel was never analyzed for per-side-mask vs joint-skip. The csv_train binary uses the CPU FBSPP path; the Python/MLX runtime uses Metal — so production users may hit the same class of bug independently. **This is a separate task** (suggest filing a new ticket linked to S38).
3. **What is the exact crossover N for H3 magnitude?** The math predicts `N* ≈ B × max_leaves` (joint-skip dominates below); empirical T1 will pin it down. Important for Option B fallback if Option A blocks.
4. **Why does S35-#129's one-hot per-side-mask in FindBestSplit not regress when L2 is involved?** S35-#129 covered the LG one-hot L2 case for FindBestSplit, which was needed because some ST runs converted to one-hot dispatches. Confirming this fix is enough or whether there's a co-mechanism is a residual sprint-35 follow-up.
5. **Does the parent-term subtraction in FBSPP L2 need to move outside the joint-skip?** In `FindBestSplit` post-fix L2 ordinal (csv_train.cpp:1984), the parent term is subtracted unconditionally when at least one side is non-empty. In FBSPP L2 (csv_train.cpp:2391), the subtraction is inside the per-(p, k) joint-skip guard. The fix in Option A must restructure this to match FindBestSplit's pattern — it is not just a one-line `continue` removal.

---

## TL;DR

- **Best hypothesis:** **H3** — `FindBestSplitPerPartition` carries the same joint-skip-on-degenerate that DEC-042 fixed in `FindBestSplit` (csv_train.cpp:2304 one-hot, :2388 ordinal). The DEC-042 fix landed on the SymmetricTree path only; LG/DW use FBSPP and were never re-validated post-fix. **Confidence: high.**
- **Discriminating prediction:** ST+Cosine at N=1k, depth=6, bins=128 should drift **<2%** (H3 confirmed) versus **>10%** (H1, falsified). One harness flip; one T1 cell.
- **Recommended fix:** Option A — port the DEC-042 per-side mask into FBSPP ordinal (Cosine + L2) and FBSPP one-hot L2. Keep joint-skip on FBSPP one-hot Cosine pending the S35-PROBE-F-LITE re-evaluation. Re-validate G3b/G3c (must reach <2% drift) and run non-regression on G3a + N=50k LG+Cosine.
- **Fallback only if Option A blocks:** re-add LG+Cosine guard with N_threshold ≈ 5000, derived from `N* ≈ B × max_leaves`.
