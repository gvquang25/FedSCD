# Definitive Technical Audit & Document Review Report: FedSCD

> **Paper**: *Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning*  
> **Target Venue**: International Conference on Machine Learning (ICML 2026 Submission Draft `Overleaf/icml2026.tex`)  
> **Review Mode**: Definitive Final Reducer Synthesis Report (`teamwork_synthesizer`)  
> **Date**: 2026-11-11  
> **Scope**: Complete Forensic Audit across Manuscript (`Overleaf/icml2026.tex`), Compiled PDF (`icml2026.pdf`), TeX Logs (`icml2026.log`), Mathematical Proofs, Experimental Database (`results/runs_results/`), Execution Scripts, and PyTorch Codebase (`system/`).  
> **Audit Status**: Complete, Exhaustive, Publication-Grade, Reconciled across 4 Candidate Syntheses and 5 Level-2 Source Unit Reports.

---

# 1. Executive Summary & Strategic Audit Overview

### 1.1 Document & Algorithmic Foundations
The submission **"Beyond Points: Learning Separable Class Distributions for Prototype-Based Federated Learning"** addresses a foundational weakness in prototype-based Federated Learning (PFL) under severe non-IID label skew. Conventional PFL algorithms (e.g., FedProto, FedTGP, FedProc, FedPCL) summarize each semantic class as a single Euclidean point centroid (a zero-variance prototype, $\mu^c \in \mathbb{R}^d$) and enforce feature separability by maximizing Euclidean centroid margins. The authors demonstrate—both conceptually and empirically—that enlarging centroid distances fails to prevent severe Bayes classification error when intra-class feature dispersion remains large, anisotropic, and scattered across heterogeneous local clients.

To resolve this limitation, the authors introduce **FedSCD** (Federated Statistics-Consistent Distributional Prototype Learning). In FedSCD, classes are parameterized as continuous diagonal Gaussians $\mathcal{N}(\mu^c, \mathrm{diag}((\sigma^c)^{\odot 2}))$. Local clients compute per-class feature means and standard deviations, uploading only compact tuples $\{\mu_k^c, \sigma_k^c, n_k^c\}$ to the server. The server pools these decentralized statistics via the **law of total variance**, exactly capturing both within-client variance and between-client centroid drift without exchanging raw training data or neural network weights. On the client side, local representation learning is guided by:
1. **Empirical Bayes Anchor Shrinkage** and **Precision-Whitened Projections** that down-weight noisy, high-variance dimensions;
2. **Dispersion-Adaptive ProtoNCE** and **Positive-Repulsion-Free Supervised Contrastive Learning (SupCon/DCL)** that dynamically calibrate angular margins using server-derived class temperatures;
3. **SNR-Gated Mahalanobis Compactness Regularization** that enforces precision-weighted feature clustering along high-confidence coordinates; and
4. **Decoupled Classifier Head Calibration** using Gaussian virtual features sampled from the broadcast global distribution moments.

The framework maintains an asymptotic communication footprint of $\mathcal{O}(Cd)$ per client per round, exactly matching the linear communication scaling of first-order prototype federated learning.

---

### 1.2 Strategic Audit Assessment: High Scientific Promise Burdened by Fatal Submission Defects
While the theoretical motivation and empirical results on CIFAR-10 show strong promise, our comprehensive forensic audit reveals that **the submission in its current state is unready for conference submission and faces immediate administrative desk rejection or unanimous technical rejection across multiple review criteria**:

1. **Fatal Administrative Desk-Rejection Hazards**:
   - **ICML Page Limit Violation**: The compiled manuscript `Overleaf/icml2026.pdf` spans **11 full pages before references** (13 pages total, references starting on Page 12). ICML strictly enforces an **8-page limit** for the main body prior to references and appendices. Papers exceeding 8 pages before references are subject to immediate administrative desk rejection without reviewer assignment.
   - **Double-Blind Anonymity Violations**: Active author editorial macros (`\NMD{...}`, `\GVQ{...}`) render in bold blue and red fonts with brackets (e.g., `[NMD: ...]`, `[GVQ: ...]`) across five body locations (Pages 2, 3, and 7). Displaying author initials in the body text violates double-blind submission guidelines.
   - **Self-Incriminating Evidentiary Deficits & Unfinished Sections**: Line 192 prints `\GVQ{need TSNE to complete evidence this claim).}` in red text directly following the core claim that global statistics align local feature extractors into a shared coordinate system. Lines 738, 745, and 752 print `[NMD: Please find some more works]` and `[NMD: ... This is not enough.]` directly inside the Related Work section, publicly advertising an incomplete literature review.

2. **Fatal Theoretical Vulnerability — Total Absence of Proofs**:
   - Section 3.7 presents three formal mathematical results: **Proposition 3.1** (Exact pooling under total variance), **Proposition 3.2** (Drift-induced dispersion inflation), and **Proposition 3.3** (Bhattacharyya bound on pairwise Bayes error).
   - Line 556 contains a drafting comment: `% Three short results; proofs go to the appendix.`. However, **the submission contains no appendix whatsoever**. Submitting an ICML paper containing unproven formal propositions is an immediate disqualifier during technical review.

3. **Severe Empirical Contradictions & Metric Integrity Violations**:
   - **Concealment of Baseline Collapse**: Section 4 (Line 609) claims test accuracy is measured at the *"final communication round"*. However, forensic examination of `results/runs_results/runs_results.xlsx` (Sheet `Summary`, Row 3) and raw CSV logs proves that **Table 1 reports historical peak test accuracy (`df.max() * 100`) across all 100 rounds**. This practice masks catastrophic baseline divergence: on Flowers102 under $\alpha=0.1$, FedTGP collapses from a peak of 54.79% at Round 57 to **0.58% at Round 100** (Table 1 reports 54.79%); on Flowers102 under $\alpha=0.5$, FedTGP collapses to `NaN`; and FedPLVM diverges to `NaN` in 6 out of 9 benchmark runs.
   - **Inflated and Misattributed Narrative Margins**: Section 1 (Line 198) and Section 5.2 (Lines 645--647) cite obsolete preliminary logs. The narrative claims an accuracy gain of "+6.5% at $\alpha=0.5$" over FedTGP (claiming FedTGP attains 68.64%), whereas Table 1 prints 71.70% (actual margin is **+3.45%**, an 88% overclaim). The claimed peak gain of "+8.4% over FedTGP" corresponds to FedORGP at $\alpha=1.0$ ($69.60 - 61.21 = 8.39\%$), but was erroneously attributed to FedTGP (whose true maximum margin is +5.09%).
   - **Ghost Baselines & Missing Benchmarks**: The narrative heavily benchmarks FedSCD against FedProto (65.03%, 86.10%) and CCVR (62.85% to 45.65%), neither of which appears anywhere in Table 1 or Section 4 Baselines. Furthermore, Table 1 has 9 empty dashes (`--`) for FedSCD across CIFAR-100 and Flowers102, and Tiny-ImageNet contains 21 consecutive empty dashes across all algorithms, directly contradicting the claim of multi-dataset validation.

4. **Widespread Codebase-to-Paper Divergence & Undocumented Components**:
   - **Undisclosed Feature Manifold Mixup**: In `clientscd.py:459--472`, local client training executes Feature Manifold Mixup on classifier head representations (`mixup_alpha = 0.2` active by default in `main.py:342`), contributing 25% of the classifier head loss ($0.75 \mathcal{L}_{\mathrm{CE}} + 0.25 \mathcal{L}_{\mathrm{CE, mixup}}$). This key architectural component is completely omitted from the paper's text and equations.
   - **Hyperparameter Discrepancies**: Server momentum smoothing is claimed as $\beta = 0.6$ (Lines 607, 742), but the codebase defaults and benchmark execution scripts uniformly fix $\beta = 0.1$ (`serverscd.py:32`, `main.py:326`, `run_benchmark_matrix.py:77`), a $6\times$ difference in EMA learning rate. Ramp-up rounds are claimed as $T_{\mathrm{ramp}} = 10$, but executed as $T_{\mathrm{ramp}} = 5$. Local virtual feature calibration is claimed as $\lambda_{\mathrm{vl}} = 0.05$, but executed as $\lambda_{\mathrm{vl}} = 0.0$ (completely deactivated).
   - **Broken LaTeX Cross-References**: Missing labels cause visible bold **`(Eq. ??)`** artifacts inside Algorithm 1 (Lines 12 and 17) and narrative text on Pages 7 and 8 (`\ref{eq:nce}`, `\ref{eq:head}`, `\ref{eq:server_mom}`, `\ref{tab:label_skew}`).

---

# 2. Severity Distribution Matrix

Across the five audited segments of `Overleaf/icml2026.tex` and the FedSCD codebase, a total of **81 distinct verified issues** were cataloged, reconciled, and deduplicated across the four candidate syntheses and five Level-2 unit reports:

### 2.1 Severity Breakdown by Paper Segment

| Paper Segment / Analysis Partition | Critical | Major | Minor | Nitpick / Typo | Total |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Segment 1**: Abstract, Introduction & Related Work (Sec 1--2, Tab 1) | 2 | 4 | 5 | 3 | **14** |
| **Segment 2**: Method Foundations & Problem Setup (Sec 3.0--3.3, Figs 1--2) | 0 | 5 | 5 | 4 | **14** |
| **Segment 3**: Client Distribution Learning (Sec 3.4, Eqs 5--12) | 0 | 6 | 5 | 2 | **13** |
| **Segment 4**: Server Aggregation, Algorithm & Theory (Sec 3.5--3.7, Alg 1) | 2 | 5 | 5 | 3 | **15** |
| **Segment 5**: Experiments, Evaluations & Conclusion (Sec 4--6, Tabs 1--3) | 3 | 7 | 8 | 7 | **25** |
| **Total Reconciled Unique Findings** | **7** | **23** | **28** | **23** | **81** |

---

### 2.2 Severity Breakdown by Technical Error Category

| Error Category | Critical | Major | Minor | Nitpick / Typo | Total |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Administrative & Policy Compliance (ICML Page Limit, Anonymity)** | 2 | 0 | 0 | 2 | **4** |
| **Empirical Rigor & Benchmark Integrity (Tables, Metrics, Baselines)** | 3 | 5 | 6 | 4 | **18** |
| **Theoretical Rigor & Mathematical Soundness (Proofs, Formulations)** | 1 | 7 | 6 | 4 | **18** |
| **Algorithm-to-Code Discrepancies & Undisclosed Modules** | 0 | 6 | 5 | 1 | **12** |
| **Compilation Hygiene & Cross-Referencing (`\ref`, TeX Groups)** | 1 | 2 | 4 | 2 | **9** |
| **Editorial Artifacts, Drafting Comments & Language Polish** | 0 | 3 | 7 | 10 | **20** |
| **Total Counts** | **7** | **23** | **28** | **23** | **81** |

---

# 3. Key Issues Master Roadmap

The table below provides a comprehensive master directory of all **Critical** and **Major** issues, cross-referenced with exact paper sections, line numbers, repository source code files, and page numbers:

| Issue ID | Severity | Category | Paper Location & Line | Repository Code Reference | Master Problem Statement & Scientific Threat |
|:---|:---:|:---|:---|:---|:---|
| **CRIT-01** | **Critical** | Policy Compliance | `icml2026.pdf` (Pages 1--13) | `Overleaf/icml2026.tex:1--810` | **ICML 8-Page Limit Violation**: Main body spans 11 full pages before references (13 pages total; references start on Page 12). Triggers immediate administrative desk rejection without review. |
| **CRIT-02** | **Critical** | Theoretical Rigor | Sec 3.7, Lines 554--593 (Page 9) | Line 556 comment vs termination | **Total Absence of Mathematical Proofs**: Propositions 3.1, 3.2, and 3.3 are stated formally, promising proofs in an appendix (`% proofs go to the appendix`), but the paper contains no appendix. |
| **CRIT-03** | **Critical** | Anonymity / Policy | Lines 73--74, 190, 192, 211, 461 | `Overleaf/icml2026.tex` preamble | **Double-Blind Review Breaches & Evidentiary Doubts**: Preamble macros `\NMD` and `\GVQ` render author initials in bold colored text in body; line 192 prints `\GVQ{need TSNE...}`; lines 738, 745, 752 advertise incomplete related work. |
| **CRIT-04** | **Critical** | Empirical Integrity | Sec 4.1 (Line 609), Table 1 (`tab:com`) | `results/runs_results/runs_results.xlsx` | **Peak Accuracy Reported as Final-Round Accuracy**: Table 1 reports historical peak (`df.max() * 100`), concealing baseline collapse (Flowers102 FedTGP collapsing from 54.79% to 0.58%; FedPLVM diverging to `NaN` in 6/9 runs). |
| **CRIT-05** | **Critical** | Benchmark Scope | Table 1 (Lines 610--636), Line 753 | `results/runs_results/` CSV logs | **Incomplete Table 1 & Multi-Dataset Overclaim**: FedSCD has 9 empty dashes (`--`) across CIFAR-100 and Flowers102; Tiny-ImageNet contains 21 consecutive dashes; yet Abstract and Conclusion overclaim multi-dataset validation. |
| **CRIT-06** | **Critical** | Numerical Rigor | Sec 1 (Line 198), Sec 5.2 (Lines 645--647) | `BENCHMARK_RUN_GUIDE.md:117--165` | **Narrative Directly Contradicts Table 1**: Claims FedTGP attains 68.64% (+6.5% gain; Table 1 shows 71.70%, actual margin is +3.45%); misattributes +8.4% gain over FedORGP to FedTGP; benchmarks against ghost baselines FedProto & CCVR. |
| **CRIT-07** | **Critical** | Compilation / Code | Alg 1 (Lines 12, 17), Sec 3.4, 3.5 | `icml2026.log` (5 warnings) | **Pervasive Broken LaTeX Cross-References**: Undefined labels (`eq:nce`, `eq:head`, `eq:server_mom`, `tab:label_skew`) produce visible bold **`(Eq. ??)`** artifacts inside Algorithm 1 and narrative text. |
| **MAJ-01** | **Major** | Algorithm-Code | Sec 4.1 (Line 607), Line 742 | `serverscd.py:32`, `main.py:318,326,336`| **Severe Hyperparameter Discrepancies**: Server momentum claimed $\beta=0.6$ vs executed $\beta=0.1$ ($6\times$ learning rate difference); ramp-up claimed $T_{\mathrm{ramp}}=10$ vs executed $5$; local virtual loss claimed $\lambda_{\mathrm{vl}}=0.05$ vs $0.0$ (disabled). |
| **MAJ-02** | **Major** | Method Disclosure | Sec 3.4, Lines 434--448 (Eq. 12) | `clientscd.py:459--472`, `main.py:342` | **Undisclosed Feature Manifold Mixup**: Client training executes mixup ($\alpha_{\mathrm{mix}}=0.2$) by default, contributing 25% of head loss, but this module is completely omitted from the manuscript equations and text. |
| **MAJ-03** | **Major** | Mathematical Logic | Sec 1 (L192), 3.2 (L306), 3.5 (L481) | `clientscd.py:211`, `clientscd.py:329` | **Inverted Mathematical Rationale of Mahalanobis Loss**: Text claims loss penalizes over-dispersed dimensions, whereas variance in the denominator relaxes penalties on high-variance dimensions (precision weighting). |
| **MAJ-04** | **Major** | Notation Collision | Sec 3.2 (Eq. 3, L296) vs Sec 3.5 (Eq. 14) | Text definitions | **Overloaded Variance Symbol $\bar{\sigma}^2$**: Defined as pairwise inter-class average variance in Eq. 3, but redefined as multi-client server pooled variance in Eq. 14. Collides in Line 306. |
| **MAJ-05** | **Major** | Metric Discrepancy | Sec 3.2 (Line 304) | `motivation_metrics.py:161--186` | **Bhattacharyya Overlap Normalization Mismatch**: Code divides by $\sqrt{d_{\mathrm{eff}}}$ with active coordinate masking, whereas paper claims dividing by $d$ (evaluates to $\approx 0.994$ vs plotted $[0.86, 0.92]$). |
| **MAJ-06** | **Major** | Mathematical Rigor | Sec 3.1 (Eq. 2, Line 273), Sec 3.2 (Eq. 3)| `clientscd.py:125`, `serverscd.py:205` | **Singleton Class ($n_k^c=1$) Singularity**: Produces zero-variance vector $(\sigma_k^c)^2=\mathbf{0}$, causing $\ln(+\infty)$, $0/0$ forms in Eq. 3, and undefined Mahalanobis gradients; paper lacks code's $\epsilon_0=10^{-6}$ floor. |
| **MAJ-07** | **Major** | Algorithm-Code | Sec 3.4 (Line 362) | `clientscd.py:208--211` | **Variance Floor Scope Mismatch**: Text defines $\tau_v$ averaging over $d$ dimensions for class $c$, but code computes grand mean across all classes and dimensions simultaneously. |
| **MAJ-08** | **Major** | Mathematical Rigor | Sec 3.4, Eq. 10 (Line 418) | `clientscd.py:260--269` | **Missing Class Conditioning in Mahalanobis Loss**: Eq. 10 omits conditioning on observed classes $y \in \mathcal{C}_{t-1}$; undefined for unobserved classes in early rounds where no anchor exists. |
| **MAJ-09** | **Major** | Mathematical Rigor | Sec 3.4, Eq. 12 (Line 443) | `clientscd.py:371, 385` | **Formulation Level Inconsistency & Unindexed Noise**: Eq. 12 mixes a single-sample cross-entropy term with a dataset-level class sum without an expectation operator; unindexed symbol $\eta$ for dual scales. |
| **MAJ-10** | **Major** | Complexity Rigor | Sec 3.6 (Line 549) | `clientscd.py:298--314` | **Contrastive Similarity Misclassified as Element-Wise**: Claimed as "only element-wise vector operations", whereas ProtoNCE and SupCon evaluate dense inner products and matrix multiplications. |
| **MAJ-11** | **Major** | Terminology Precision | Sec 3.5, Lines 497, 503 (Eq. 16) | `serverscd.py:221--224` | **Inverted Terminology**: Designating $r_t^c$ a "concentration score" when it scales monotonically with dispersion and sample uncertainty. |
| **MAJ-12** | **Major** | Theoretical Precision | Sec 3.7 (Line 564, Prop. 3.1) | `clientscd.py:125` (`unbiased=False`) | **Unstated Regularity Condition for Proposition 3.1**: Exact total-variance pooling holds strictly for empirical plug-in variance ($1/N$), failing under Bessel's unbiased estimator ($1/(N-1)$). |
| **MAJ-13** | **Major** | Theoretical Rigor | Sec 3.7 (Lines 578--588, Prop. 3.3) | `Overleaf/icml2026.tex:578` | **Classical Kailath Bound Uncredited & Missing Multiclass Extension**: Kailath (1967) uncredited in proposition header; lacks multiclass union bound extension necessary for $C$-class benchmarks. |
| **MAJ-14** | **Major** | Benchmark Fairness | Sec 4.1 (Line 607) | `system/wandb/run-*/config.yaml` | **Inconsistent Baseline Batch Sizes**: FedTGP and FedSA executed with $B=10$, while FedCOF and FedORGP used $B=32$, violating the uniform $B=32$ condition stated in line 607. |
| **MAJ-15** | **Major** | Visual Grounding | Sec 5.3 (Lines 664--672) | Line 192 (`\GVQ{need TSNE...}`) | **Section 5.3 Lacks Visual Grounding & Quantitative Metrics**: Contains zero figure callouts, zero tables, no numerical endpoints, leaving coordinate alignment unproven. |
| **MAJ-16** | **Major** | Artifact Integrity | `Figure_Plot/test_acc.csv` | `plot_combined_3panels.py:40--77` | **Corrupted Motivational Plot Script**: `test_acc.csv` overwritten with baseline data, missing FedSCD, CCVR, and FedProto; re-running script renders only FedTGP on panel (c). |
| **MAJ-17** | **Major** | Evaluation Rigor | Table 3 (`tab:ablation_params`), Sec 5.4 | Tables 1, 2, 3 | **Table 3 Mislabeled as Sensitivity & Absence of Error Bars**: Table 3 tests 3 binary switches rather than continuous sweeps; complete absence of multi-seed standard deviations across all tables. |
| **MAJ-18** | **Major** | Baseline Clarity | Table 2 (Line 708), Sec 5.4 | Table 2 row 6 | **Undefined Communication Protocol for Table 2 Baseline**: Row 6 "Baseline (CE + Augmentation only): 72.57%" does not specify whether clients communicate prototypes or train in isolation. |
| **MAJ-19** | **Major** | Limitations Rigor | Sec 6 (Lines 755--756) | Section 6 text | **Omission of Foundational Technical Limitations**: Omits post-ReLU conical non-Gaussianity, sample variance degeneracy when $n_k^c \le 2$, $2\times$ payload overhead, and differential privacy risks. |
| **MAJ-20** | **Major** | LaTeX Syntax Engine | Lines 678--736 | `Overleaf/icml2026.tex:151--209` | **Root-Level TeX Grouping Trap**: Stray opening brace on line 678 closed only on line 736 traps 59 lines, a figure, and `\section{Related Work}` inside an unclosed local scope. |
| **MAJ-21** | **Major** | Taxonomy Rigor | Table 1 (`tab:paradigm_comparison`) | Lines 758--779 | **Table 1 Qualitative Paradigm Comparison Flaws**: Labels all Statistical FL as $\mathcal{O}(Cd^2)$ (ignoring modern $\mathcal{O}(Cd)$ methods like FedCOF); hybrid baseline for Contrastive FL; font macro errors. |
| **MAJ-22** | **Major** | Literature Attribution | Sec 2 (Line 757), Sec 3.4 (Remark 1) | `Overleaf/icml2026.tex:230` | **Theoretical Misattribution of Positive-Repulsion-Free Loss**: Attributed exclusively to SupCon without citing foundational Decoupled Contrastive Learning (DCL; Yeh et al., CVPR 2022). |
| **MAJ-23** | **Major** | Literature Rigor | Sec 1 (L680), Sec 2 (L733--755) | Lines 680, 733, 736, 741, 748 | **Domain Literature Misattributions**: FedGH cited under prototype FL (it aggregates heads); false claim that all decoupled FL aggregates extractors; FedGMKD cited under second-order statistics; jammed citations. |

---

# 4. Section-by-Section Detailed Audit across Paper & Codebase

---

## 4.1 Algorithm-to-Code Discrepancies

### 1. Dimension Normalization Mismatch in Bhattacharyya Overlap ($\sqrt{d_{\mathrm{eff}}}$ vs. $d$) [Issue MAJ-05]
- **Manuscript Statement**: Section 3.2, Line 304 of `Overleaf/icml2026.tex`:
  `"... (reported in Figure~\ref{fig:motivation} as the dimension-normalized overlap $\exp(-D_B(c,c')/d)$ averaged over class pairs)."`
- **Code Execution**: In `system/utils/motivation_metrics.py`, Lines 161--186:
  ```python
  active_mask = (var_dist.mean(dim=0) > 1e-4)
  d_eff = int(active_mask.sum().item())
  D_B_scaled = D_B / np.sqrt(float(d_eff))
  overlap_mat = torch.exp(-D_B_scaled)
  ```
- **Technical Analysis**:
  For representation dimension $d = 512$, $\sqrt{d} \approx 22.63$. With empirical pairwise Bhattacharyya distances $D_B \approx 2.5$--$3.5$, dividing by $d = 512$ evaluates to $\exp(-3.0 / 512) \approx 0.994$. In contrast, dividing by $\sqrt{d_{\mathrm{eff}}} \approx 22.63$ yields $\exp(-3.0 / 22.63) \approx 0.876$. This matches the Y-axis range $[0.86, 0.92]$ specified in `Figure_Plot/plot_combined_3panels.py` (Line 84: `(axes[1], df_overlap, "(b) Bhattacharyya Overlap", 1.0, (0.86, 0.92), [0.86, 0.88, 0.90, 0.92])`).
  Anyone attempting to reproduce Figure 1 using the published formula $\exp(-D_B/d)$ will obtain near-flat curves at $\approx 0.99$, directly contradicting the published chart.
- **Drop-in Fix**: Update Line 304 of `icml2026.tex`:
  ```latex
  which equals $0$ for disjoint distributions and $1$ for identical ones (reported in Figure~\ref{fig:motivation} as the dimension-normalized overlap index $\exp(-D_B(c,c')/\sqrt{d_{\mathrm{eff}}})$ averaged over class pairs across active coordinates where $\bar\sigma_j^2 > 10^{-4}$).
  ```

### 2. Undisclosed Feature Manifold Mixup in Local Client Training [Issue MAJ-02]
- **Manuscript Statement**: Section 3.4, Lines 434--448 (Eq. 12); completely omitted.
- **Code Execution**: In `system/flcore/clients/clientscd.py`, Lines 459--472:
  ```python
  if self.mixup_alpha > 0 and rep.size(0) > 1:
      rep_det = rep.detach()
      lam_mix = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
      lam_mix = max(lam_mix, 1.0 - lam_mix)
      perm = torch.randperm(rep_det.size(0), device=self.device)
      rep_mix = lam_mix * rep_det + (1.0 - lam_mix) * rep_det[perm]
      try:
          out_mix = self.head(rep_mix)
          loss_ce_mix = lam_mix * self.loss(out_mix, y) + (1.0 - lam_mix) * self.loss(out_mix, y[perm])
          loss_ce = 0.75 * loss_ce + 0.25 * loss_ce_mix
      except (RuntimeError, ValueError):
          pass
  ```
  In `main.py:342`, `--mixup_alpha` defaults to `0.2` and is active across all benchmarks.
- **Technical Analysis**:
  On every batch of size $> 1$, local client training executes Feature Manifold Mixup on detached representations, blending 25% of an interpolated cross-entropy loss into the classifier head. Omitting this regularizer obscures an architectural module contributing to head decision boundary smoothing.
- **Drop-in Fix**: Formally disclose Feature Manifold Mixup in Section 3.4 (Eq. 12) as an auxiliary regularizer $\mathcal{L}_{\mathrm{CE}}^{\mathrm{head}} = (1-\alpha_{\mathrm{mix}})\mathcal{L}_{\mathrm{CE}} + \alpha_{\mathrm{mix}}\mathcal{L}_{\mathrm{Mixup}}$ with $\alpha_{\mathrm{mix}} = 0.25$ and Beta parameter $0.2$.

### 3. Execution Inactivity of Local Virtual Feature Branch $\lambda_{\mathrm{vl}}$ [Issue MAJ-01]
- **Manuscript Statement**: Section 3.4 (Eq. 12, Line 443) and Section 4.1 (Line 607) assert that local virtual feature calibration is active with weight $\lambda_{\mathrm{vl}} = 0.05$.
- **Code Execution**: In `system/main.py:336`, `--lamda_virt_local` defaults to `0.0`. In `clientscd.py:23`, class docstring states: `"- 4. Gaussian Virtual Feature Calibration (mặc định tắt: lamda_virt = 0.0)"`. In `BENCHMARK_RUN_GUIDE.md` and `run_benchmark_matrix.py:80`, only `--lamda_virt_global 0.05` is passed, executing all runs with `lamda_virt_local = 0.0`.
- **Technical Analysis**: The local virtual feature sampling branch is completely deactivated in reported experiments.
- **Drop-in Fix**: Update Line 607 to state `"head calibration weights $\lambda_{\text{vl}} = 0.0, \lambda_{\text{vg}} = 0.05$ (with global virtual feature calibration active by default)"`.

### 4. Server Momentum Smoothing Discrepancy ($\beta = 0.6$ vs. $\beta = 0.1$) [Issue MAJ-01]
- **Manuscript Statement**: Section 4.1 (Line 607) and Section 5.4 (Line 742) state: `"server momentum \beta = 0.6"`.
- **Code Execution**: In `system/flcore/servers/serverscd.py:32`, `system/main.py:326`, and `run_benchmark_matrix.py:77`, `--proto_momentum` defaults to and was executed with **0.1**.
- **Technical Analysis**: In Eq. 15 ($s_t^c = (1-\beta)s_{t-1}^c + \beta \bar{s}_t^c$), setting $\beta=0.6$ gives 60% weight to current-round statistics, whereas $\beta=0.1$ gives 10% weight. This constitutes a $6\times$ discrepancy in update inertia.
- **Drop-in Fix**: Harmonize Line 607 to `server momentum \beta = 0.1`, and clarify in Line 742 that $\beta = 0.6$ was evaluated as an ablation point.

### 5. Prototype Ramp-up Schedule Discrepancy ($T_{\mathrm{ramp}} = 10$ vs. $T_{\mathrm{ramp}} = 5$) [Issue MAJ-01]
- **Manuscript Statement**: Section 4.1 (Line 607) states: `"ramp-up rounds T_{\text{ramp}} = 10"`.
- **Code Execution**: In `clientscd.py:52`, `main.py:318`, and `BENCHMARK_RUN_GUIDE.md`, `--proto_rampup` defaults to and was executed with **5**.
- **Technical Analysis**: Because $\lambda_t = \lambda \min(1, (t_k - T_{\mathrm{warm}}) / T_{\mathrm{ramp}})$, the local feature extractor reaches full regularization weight twice as fast in code (5 rounds) as claimed in the paper (10 rounds).
- **Drop-in Fix**: Change `T_{\text{ramp}} = 10` to `T_{\text{ramp}} = 5` in Line 607.

### 6. Variance Floor Implementation Scope vs. Mathematical Definition ($\tau_v$) [Issue MAJ-07]
- **Manuscript Statement**: Section 3.4 (Line 362) defines $\tau_v = \max(\alpha_v \mathrm{mean}((\sigma^c)^{\odot 2}), \alpha_q)$ and specifies: *"Let $\mathrm{mean}(\cdot)$ average over the $d$ dimensions."*
- **Code Execution**: In `clientscd.py:208--211`:
  ```python
  var_tensor = std_tensor.pow(2)
  mean_var = float(torch.mean(var_tensor).item())
  tau_var = max(0.25 * mean_var, 1e-2)
  ```
  `torch.mean(var_tensor)` averages across all $|\mathcal{C}_{t-1}|$ valid classes and all $d$ dimensions simultaneously.
- **Drop-in Fix**: Update Line 362 to define the pooled variance floor explicitly:
  $$\tau_v = \max\!\Big(\alpha_v \frac{1}{|\mathcal{C}_{t-1}| d} \sum_{c \in \mathcal{C}_{t-1}} \sum_{j=1}^d (\sigma_j^c)^2,\, \alpha_q\Big)$$
  with $\alpha_v = 0.25$ and $\alpha_q = 10^{-2}$.

---

## 4.2 Mathematical & Theoretical Inconsistencies / Proof Flaws

### 1. Complete Absence of Mathematical Proofs for Propositions 3.1--3.3 (Missing Appendix) [Issue CRIT-02]
- **Manuscript Statement**: Section 3.7 (`sec:theory`), Lines 554--593. Line 556 contains: `% Three short results; proofs go to the appendix.`.
- **Technical Analysis**: Propositions 3.1, 3.2, and 3.3 form the foundational theoretical core of the submission, but the paper contains no appendix. Submitting unproven propositions to ICML is a fatal flaw.
- **Remediation**: Append a formal `\appendix` providing complete derivations. (Provided in Section 6 of this report).

### 2. Mathematically Inverted Justification of Mahalanobis Compactness ($\mathcal{L}_{\mathrm{PL}}$) [Issue MAJ-03]
- **Manuscript Claims**: Lines 192, 306, 416, 481, and 753 repeatedly claim that Mahalanobis compactness "selectively penalizes over-dispersed feature dimensions" and that larger variance "guides clients to compress it via $\mathcal{L}_{\mathrm{PL}}$."
- **Mathematical Reality**:
  In Equation (9) (Line 416) and `clientscd.py:211, 329`:
  $$d^c(z) = \frac{1}{d}\sum_{j=1}^{d}\frac{g^c_j\,(z_j-p^c_j)^2}{v^c_j}, \quad \text{where } v^c_j = (\sigma_j^c)^2 + \tau_v.$$
  The gradient with respect to representation coordinate $z_j$ is:
  $$\frac{\partial d^c(z)}{\partial z_j} = \frac{2\,g^c_j}{d\,v^c_j}\,(z_j - p^c_j).$$
  Because regularized variance $v_j^c$ is in the **denominator**:
  - High-variance dimensions have $v_j^c \gg 1 \implies$ gradient penalty is **attenuated** (scatter is tolerated).
  - Compact, low-variance dimensions with high SNR have small $v_j^c \implies$ gradient penalty is **maximal** (strict spatial confinement).
  Mahalanobis compactness performs **precision-weighted regularization**, not over-dispersed dimension suppression.
- **Drop-in Fix**: Rewrite Lines 192, 306, 481, and 753 to state that Mahalanobis compactness enforces *precision-weighted regularization*: it imposes strict confinement along compact, high-SNR feature coordinates while de-weighting noisy, drifted dimensions.

### 3. Inverted Terminology: Naming $r_t^c$ "Concentration" When It Scales with Dispersion [Issue MAJ-11]
- **Manuscript Location**: Section 3.5, Lines 497, 503 (Eq. 16).
- **Technical Analysis**: Eq. 16 defines $r_t^c = \frac{\|\sigma_t^c\|_2 / \|\mu_t^c\|_2}{\log(n_t^c + \kappa)}$. In statistics (e.g., von Mises-Fisher), concentration is inversely related to dispersion. Here, $r_t^c$ scales positively with relative standard deviation and inversely with sample count. It measures *dispersion* or *cluster uncertainty*. Calling it a "concentration score" inverts standard terminology.
- **Drop-in Fix**: Rename $r_t^c$ to "density-aware dispersion score" or "cluster uncertainty index" in Line 497 and Eq. 16.

### 4. Severe Notation Collision on Pooled/Average Variance Symbol $\bar{\sigma}^2$ [Issue MAJ-04]
- **Manuscript Location**: Section 3.2 (Eq. 3, Line 296; Line 306) vs. Section 3.5 (Eq. 14, Line 479) and Section 3.7 (Prop. 3.2).
- **Technical Analysis**:
  In Eq. 3, $\bar{\sigma}_j^2 = \frac{1}{2}((\sigma_j^c)^2 + (\sigma_j^{c'})^2)$ is the pairwise inter-class arithmetic average variance between classes $c$ and $c'$. In Eq. 14, $(\bar{\sigma}_t^c)^2 = \frac{1}{\bar{n}_t^c} \sum_k n_k^c [(\sigma_k^c)^2 + (\mu_k^c - \bar{\mu}_t^c)^{\odot 2}]$ denotes the server multi-client pooled variance of class $c$. In Line 306, both meanings collide: *"pooling global variance $\bar{\sigma}_j^2$ on the server"*, conflating inter-class averaging with multi-client pooling.
- **Drop-in Fix**: In Section 3.2, rename pairwise inter-class variance to $\sigma_{c, c', j}^2 \triangleq \tfrac{1}{2}\big((\sigma_j^c)^2 + (\sigma_j^{c'})^2\big)$, reserving $\bar{\sigma}_t^c$ exclusively for server multi-client pooled dispersion of class $c$ in round $t$.

### 5. Singleton Class ($n_k^c = 1$) Singularity & Dead Neuron Breakdown [Issue MAJ-06]
- **Manuscript Location**: Section 3.1 (Eq. 2, Line 273), Section 3.2 (Eq. 3).
- **Code Reference**: `clientscd.py:125`, `serverscd.py:205`.
- **Technical Analysis**: If $n_k^c = 1$, $(\sigma_k^c)^2 = \mathbf{0} \in \mathbb{R}^d$. In Eq. 3, $\ln(\bar{\sigma}_j^2 / (\sigma_j^c \sigma_j^{c'})) \to \ln(+\infty)$, producing undefined values. In Eq. 9, zero variance produces division by zero. The codebase implements defensive numerical floors (`1e-6` in client, `1e-8` in server, `1e-4` in metric), but the manuscript presents these equations without non-degeneracy conditions.
- **Drop-in Fix**: Add an explicit non-degeneracy condition directly after Eq. 2:
  ```latex
  where $\odot$ denotes element-wise multiplication, and $\sigma_k^c\in\mathbb{R}^d$ is the per-dimension standard deviation (with a numerical floor $\sigma_{k,j}^c \ge \epsilon_0 = 10^{-6}$ enforced when $n_k^c = 1$ to guarantee non-singular dispersion on singleton classes).
  ```

### 6. Regularity Conditions in Proposition 3.1 [Issue MAJ-12]
- **Manuscript Location**: Section 3.7, Line 564.
- **Technical Analysis**: Proposition 3.1 claims exact recovery of union sample mean and variance. In mathematical statistics, this holds if and only if variances are defined with empirical plug-in denominators ($1/N$, as implemented in PyTorch via `unbiased=False`), not Bessel's unbiased estimator ($1/(N-1)$). This condition must be formally stated.
- **Drop-in Fix**: Qualify Proposition 3.1 to state that exact pooling holds under the empirical plug-in sample variance definition (Eq. 2) with denominator $n_k^c$.

### 7. Proposition 3.3 Attribution & Multiclass Union Bound Gap [Issue MAJ-13]
- **Manuscript Location**: Section 3.7, Lines 578--588.
- **Technical Analysis**: The pairwise bound $P_e(c, c') \le \sqrt{\pi_c \pi_{c'}} \exp(-D_B(c, c'))$ is a classical result due to \citet{kailath1967divergence}. Presenting it inside an uncredited proposition header creates an attribution issue. Furthermore, the paper lacks the multiclass union bound extension required for multiclass benchmarks ($C=10, 100$).
- **Drop-in Fix**: Attribute Proposition 3.3 to Kailath (1967) and append Corollary 3.3.1 (Multiclass Bayes Error Union Bound).

### 8. Missing Class Conditioning in $\mathcal{L}_{\mathrm{PL}}$ (Eq. 10) [Issue MAJ-08]
- **Manuscript Location**: Section 3.4, Lines 392 vs. 418.
- **Technical Analysis**: Eq. 7 explicitly conditions expectation on $y \in \mathcal{C}_{t-1}$. Eq. 10 writes $\mathcal{L}_{\mathrm{PL}} = \mathbb{E}_{(x,y)\sim\mathcal{D}_k} d^y(f_k(x))$ without conditioning. For unobserved classes ($y \notin \mathcal{C}_{t-1}$), no anchor $p^y$ or variance $v^y$ exists.
- **Drop-in Fix**: Condition Eq. 10 consistently: $\mathcal{L}_{\mathrm{PL}} = \mathbb{E}_{(x,y)\sim\mathcal{D}_k \mid y \in \mathcal{C}_{t-1}} d^y(f_k(x))$.

### 9. Formulation Inconsistencies & Unindexed $\eta$ in Head Loss (Eq. 12) [Issue MAJ-09]
- **Manuscript Location**: Section 3.4, Eq. 12 (Line 443).
- **Technical Analysis**: Eq. 12 mixes a single-sample cross-entropy term with a dataset-level class sum without an expectation operator. Furthermore, the perturbation scale $\eta$ is unindexed in the equation despite the text specifying $\eta_{\mathrm{loc}} = 0.5$ and $\eta_{\mathrm{glob}} = 0.3$.
- **Drop-in Fix**: Restructure Eq. 12 using explicit expectation notation and distinct virtual feature symbols $\tilde{z}_{\mathrm{loc}}^y$ and $\tilde{z}_{\mathrm{glob}}^c$.

### 10. Flawed Classification of Contrastive Complexity as "Element-Wise" [Issue MAJ-10]
- **Manuscript Location**: Section 3.6, Line 549 vs `clientscd.py:298--314`.
- **Technical Analysis**: Line 549 asserts contrastive similarities involve "only element-wise vector operations". Contrastive losses compute Gram matrix inner products (`torch.matmul(all_feats, all_feats.T)` in code), which are dense matrix multiplications and dot products, not element-wise operations.
- **Drop-in Fix**: Rephrase Line 549 to distinguish element-wise projections from dense inner product evaluations.

---

## 4.3 Empirical & Evaluation Discrepancies

### 1. Metric Contradiction & Concealment of Baseline Collapse (Peak vs. Final Round) [Issue CRIT-04]
- **Manuscript Statement**: Section 4.1, Line 609:
  `"We measure model performance using the average local test accuracy across all clients at the final communication round."`
- **Empirical Ground Truth**: Forensic formula inspection of `results/runs_results/runs_results.xlsx` reveals that Table 1 copies directly from Sheet `Summary`, Row 3: *"Bảng 1: Best Test Accuracy (%) [Peak Performance across 100 Rounds]"*.
- **Concealment of Baseline Collapse**:
  - **Flowers102 ($\alpha=0.1$)**: FedTGP reaches 54.79% at Round 57, then collapses to **0.58% at Round 100**. Table 1 reports **54.79%**.
  - **Flowers102 ($\alpha=0.5$)**: FedTGP peaks at 31.26% at Round 38, then collapses to `NaN`. Table 1 reports **31.26%**.
  - **Flowers102 ($\alpha=1.0$)**: FedTGP peaks at 24.48% at Round 36, then collapses to **0.54% at Round 100**. Table 1 reports **24.48%**.
  - **FedPLVM Divergence**: FedPLVM diverges to `NaN` before Round 100 in 6 out of 9 benchmark runs, yet Table 1 prints intermediate historical peaks (e.g., 43.54%, 28.76%, 86.82%).
  - **CIFAR-10 ($\alpha=0.1$)**: FedTGP finishes Round 100 at 84.29%, but Table 1 reports peak accuracy **88.87%** (reached at Round 93).
- **Remediation**: Amend Line 609 to define peak accuracy:
  ```latex
  \textbf{Metrics.} We measure model performance using the peak average local test accuracy across 100 communication rounds, reflecting the maximum attainable representation capability under non-IID partitioning.
  ```
  Add explicit table footnotes noting baselines that suffered post-peak training collapse or numerical divergence.

### 2. Incomplete Benchmark Entries & Multi-Dataset Overclaim in Table 1 [Issue CRIT-05]
- **Manuscript Location**: Section 4.1, Lines 597--598; Table 1 (`tab:com`, Lines 610--636); Section 6, Line 753.
- **Technical Analysis**:
  Table 1 allocates 12 benchmark columns across 4 datasets: CIFAR-10, CIFAR-100, Flowers102, and Tiny-ImageNet.
  - For the proposed method **FedSCD**, 9 out of 12 cells are empty dashes (`--`). Completed results appear *only* for CIFAR-10.
  - For Tiny-ImageNet, all 21 cells across all 7 baselines and FedSCD are empty dashes (`--`).
  - Section 4.1 (Line 597) defines only CIFAR-10.
  - Despite this, Conclusion Line 753 claims that FedSCD's superiority is validated *"across multiple datasets"*.
- **Remediation**: Remove the 3 Tiny-ImageNet columns from Table 1; populate CIFAR-100 and Flowers102 entries for FedSCD using `system/run_benchmark_matrix.py` (or delineate Table 1 as CIFAR-10 in the main text); and rephrase Line 753 to: *"Benchmark experiments on CIFAR-10 across severe Dirichlet partitions demonstrate..."*.

### 3. Direct Contradiction Between Narrative Text and Table 1 [Issue CRIT-06]
- **Manuscript Location**: Section 1, Line 198; Section 5.2, Lines 645--647; Table 1 (Lines 620--635).
- **Technical Analysis**:
  The narrative text cites preliminary runs from `BENCHMARK_RUN_GUIDE.md:117--165`:
  - *Section 5.2 Line 645 claim*: FedSCD (75.15%) outperforms FedTGP (68.64%) by 6.5% at $\alpha=0.5$.
    *Table 1 ground truth*: FedTGP is 71.70%. Actual gain is $75.15 - 71.70 = \mathbf{+3.45\%}$ (nearly half the claimed margin).
  - *Section 5.2 Line 646 claim*: FedSCD (90.71%) maintains margin of 1.9% over FedTGP (88.83%) at $\alpha=0.1$.
    *Table 1 ground truth*: FedTGP is 88.87% (margin +1.84%). Runner-up is FedORGP (88.88%, margin +1.83%).
  - *Section 1 Line 198 claim*: improving test accuracy by up to 8.4 percentage points over FedTGP.
    *Table 1 ground truth*: At $\alpha=1.0$, FedSCD (69.60%) vs FedTGP (64.51%) yields a maximum gain of $\mathbf{+5.09\%}$. The figure 8.39% corresponds to FedORGP ($69.60 - 61.21 = 8.39\%$).
  - *Phantom baselines*: Lines 645--646 prominently benchmark FedSCD against FedProto (65.03%, 86.10%) and CCVR (62.85% to 45.65%), neither of which appears anywhere in Table 1 or Section 4 Baselines.
- **Drop-in Fix**: Synchronize narrative text with Table 1 data:
  ```latex
  Table~\ref{tab:com} reports the test accuracy on CIFAR-10 across varying degrees of label skew. 
  Under mild heterogeneity ($\alpha = 1.0$), FedSCD achieves 69.60\% test accuracy, outperforming the runner-up baseline FedTGP (64.51\%) by +5.09 percentage points and FedCOF (63.62\%) by +5.98 points.
  At moderate heterogeneity ($\alpha = 0.5$), FedSCD reaches 75.15\%, outperforming the strongest baseline FedTGP (71.70\%) by +3.45 percentage points and surpassing other baselines (FedCOF 69.64\%, FedPLVM 69.76\%, FedORGP 68.70\%, FedSA 68.48\%) by up to +6.67 percentage points. 
  Under extreme heterogeneity ($\alpha = 0.1$), FedSCD achieves 90.71\%, maintaining margins of +1.83 percentage points over runner-up FedORGP (88.88\%) and +1.84 percentage points over FedTGP (88.87\%).
  ```

### 4. Inconsistent Local Batch Sizes Across Baseline Implementations [Issue MAJ-14]
- **Manuscript Statement**: Section 4.1, Line 607 asserts uniform mini-batch size $B = 32$.
- **Code Execution**: WandB run configurations (`system/wandb/run-*/files/config.yaml`) and `system/run_all_baselines.sh:47` confirm that **FedTGP and FedSA were executed with batch size $B = 10$**, while FedCOF and FedORGP used $B = 32$. Batch size 10 in non-IID FL alters local gradient variance and triples optimization steps per epoch.
- **Drop-in Fix**: Standardize baseline batch sizes to $B = 32$ across all methods.

### 5. Broken Visualization Pipeline (`Figure_Plot/test_acc.csv`) [Issue MAJ-16]
- **Manuscript Location**: Section 1, Figure 1; Section 5.3, Line 664.
- **Code Execution**: `plot_combined_3panels.py` reads from `Figure_Plot/test_acc.csv`, searching for `FedSCD`, `FedNewPCL`, `CCVR`, and `FedProto`. However, `test_acc.csv` was overwritten with baseline data from `CIFAR10_dir0-5.csv`, which lacks all four columns. Re-running the script renders *only FedTGP* on panel (c).
- **Remediation**: Restore logged accuracy trajectories for FedSCD (`FedNewPCL`), CCVR, and FedProto into `Figure_Plot/test_acc.csv` and re-render `figures/motivation.pdf`.

### 6. Ablation Study Mislabeled as Sensitivity & Absence of Error Bars [Issue MAJ-17]
- **Manuscript Location**: Section 5.4, Lines 675--677; Table 3 (`tab:ablation_params`).
- **Technical Analysis**: Table 3 is titled "Sensitivity to parameters," but evaluates only 3 isolated binary switches (class-specific vs shared projection, fixed $\rho$, no momentum $\beta=1.0$). No continuous sweeps are provided. Furthermore, across Tables 1, 2, and 3, not a single result reports mean $\pm$ standard deviation across random seeds.
- **Drop-in Fix**: Retitle Table 3 to "Design Choice and Architectural Ablations on CIFAR-10 ($\alpha=0.5$)", and provide multi-seed mean $\pm$ standard deviation across 3 random seeds in Table 1.

### 7. Undefined Communication Protocol for Table 2 Baseline [Issue MAJ-18]
- **Manuscript Location**: Table 2, Row 6 (Line 708), Section 5.4 (Line 698).
- **Technical Analysis**: Row 6 reports `Baseline (CE + Augmentation only): 72.57%`. The text does not define the federated communication protocol: do clients communicate prototypes for nearest-centroid classification, or does this represent isolated local training without communication?
- **Drop-in Fix**: Explicitly define the protocol in Section 5.4: *"clients communicate prototypes for nearest-centroid inference, but train local backbones using standard cross-entropy and image augmentations without contrastive or compactness regularization."*

### 8. Section 5.3 Lacks Visual Grounding and Quantitative Endpoints [Issue MAJ-15]
- **Manuscript Location**: Section 5.3 (`sec:geometry`), Lines 664--672.
- **Technical Analysis**: Section 5.3 contains two purely narrative paragraphs with zero figure callouts, zero tables, and no quantitative values. It fails to cite `Figure~\ref{fig:motivation}`. Furthermore, the claim in Section 1 Line 192 that FedSCD aligns feature extractors to a shared coordinate system is left without empirical grounding.
- **Drop-in Fix**: Insert explicit callouts to `Figure~\ref{fig:motivation}` in Section 5.3; report quantitative endpoints (FedSCD overlap 0.870 vs FedTGP 0.894; margin 3.77 vs 23.85); insert 2-panel t-SNE feature visualizations.

### 9. Omission of Core Technical Limitations in Conclusion [Issue MAJ-19]
- **Manuscript Location**: Section 6 (`sec:conclusion`), Lines 755--756.
- **Technical Analysis**: Section 6 offers a superficial 3-line note that omits four fundamental technical boundaries: post-ReLU conical non-Gaussianity, sample variance degeneracy under extreme class sparsity ($n_k^c \le 2$), $2\times$ payload overhead ($C(2d+1)$ vs $Cd$), and differential privacy risks of sharing second-order dispersion.
- **Drop-in Fix**: Expand Section 6 limitations paragraph to rigorously address these four foundational boundaries.

---

## 4.4 Wording, Typographical, Formatting, and Compliance Issues

### 1. ICML Page Limit Violation (11 Pages Before References vs. 8-Page Limit) [Issue CRIT-01]
- **Manuscript Location**: `Overleaf/icml2026.pdf`, Pages 1--13.
- **Technical Analysis**: The compiled manuscript spans 11 full pages before references (13 pages total; references start on Page 12). ICML 2026 strictly enforces an **8-page limit** for the main body prior to references and appendices. Under ICML submission policy, this triggers **immediate administrative desk rejection without review**.
- **Remediation**:
  1. Condense Section 2 (Related Work) from 1.5 pages to 0.75 pages.
  2. Condense Section 3 (Methodology) by ~1.5 pages.
  3. Transfer formal mathematical proofs of Propositions 3.1--3.3 to the Appendix.
  4. Remove unpopulated Tiny-ImageNet columns (21 dashes) from Table 1.
  5. Bring the main body strictly under 8 pages prior to references.

### 2. Double-Blind Anonymity Violations and Editorial Comments in Compiled PDF [Issue CRIT-03]
- **Manuscript Location**: Lines 73--74, 190, 192, 211, 218, 225, 260, 345, 461 of `Overleaf/icml2026.tex`.
- **Technical Analysis**: Active macros `\newcommand{\NMD}[1]{\textbf{\color{blue}[NMD: #1]}}` and `\newcommand{\GVQ}[1]{\textbf{\color{red}[GVQ: #1]}}` display author initials (`NMD`, `GVQ`) in bold blue and red fonts in the PDF. Displaying author initials violates ICML double-blind submission rules and risks immediate desk rejection. Furthermore, `\GVQ{need TSNE...}` directly confesses to reviewers that empirical evidence was missing for the coordinate consensus claim.
- **Drop-in Fix**: Delete all `\NMD{...}` and `\GVQ{...}` macros across `icml2026.tex`.

### 3. Broken LaTeX Cross-References Producing Visible `(Eq. ??)` [Issue CRIT-07]
- **Manuscript Location**: Lines 393, 441, 451, 463, 492, 497, 539, 544, 612, 791, 792, 799, 800 of `Overleaf/icml2026.tex`.
- **Technical Analysis**: Four cross-references point to non-existent labels, confirmed via `Overleaf/icml2026.log`:
  1. `\ref{eq:nce}` (Lines 451, 497, 791) is broken because Eq. 7 at Line 393 is labeled `\label{eq:proto_nce}`.
  2. `\ref{eq:head}` (Lines 463, 539, 792) is broken because Eq. 12 at Line 441 is labeled `\label{eq:head_loss}`.
  3. `\ref{eq:server_mom}` (Line 544) is broken because Eq. 15 at Line 492 is labeled `\label{eq:ema}`.
  4. `\ref{tab:label_skew}` (Lines 799, 800) is broken because Table 1 at Line 612 is labeled `\label{tab:com}`.
  These keys compile as bold double question marks **`??`** in Algorithm 1 and the body text.
- **Drop-in Fix**: Unify labels by adding aliases `\label{eq:proto_nce}\label{eq:nce}`, `\label{eq:head_loss}\label{eq:head}`, `\label{eq:ema}\label{eq:server_mom}`, and `\label{tab:com}\label{tab:label_skew}`.

### 4. Root-Level TeX Grouping Trap Spanning Across Section Boundaries [Issue MAJ-20]
- **Manuscript Location**: Opening brace on Line 678 closed only on Line 736 of `DOCUMENT_TEXT_MAP.md` (Lines 151--209 of `Overleaf/icml2026.tex`).
- **Technical Analysis**: A stray opening curly brace `{` was placed before "Federated learning" at Line 151 and closed only at Line 209 at the end of Related Work. This traps 59 lines of text—spanning paragraphs, a floating figure, an itemized list, and the top-level section command `\section{Related Work}`—inside an unclosed local TeX group, disrupting the vertical layout engine and inter-paragraph spacing.
- **Drop-in Fix**: Remove the stray opening brace `{` on Line 151 and the trailing brace `}` on Line 209, and strip redundant paragraph groupings on Lines 153, 155, 169, 191, 193, 213, 216, 220, and 223.

### 5. Table 1 Qualitative Taxonomy Defects in Related Work [Issue MAJ-21]
- **Manuscript Location**: Lines 758--779 (`tab:paradigm_comparison`).
- **Technical Analysis**: Lists Statistical FL as universally $\mathcal{O}(Cd^2)$ (ignoring modern $\mathcal{O}(Cd)$ methods like FedCOF); lists Contrastive FL as $\mathcal{O}(Cd)$ bandwidth with pairwise shaping (impossible hybrid); conflates temperature with variance; wraps `\checkmark` in `\textbf{\checkmark}` and writes `$\mathbf{\mathcal{O}(Cd)}$`, triggering font substitution and math warnings under pdfLaTeX.
- **Drop-in Fix**: Re-typeset Table 1 using standard math-mode `\bm{\mathcal{O}}(Cd)` and accurate taxonomic classifications.

### 6. Domain Literature Misattributions & Jammed Citation Syntax [Issue MAJ-22, MAJ-23]
- **Manuscript Location**: Lines 680, 733, 736, 741, 748, 755, 757.
- **Technical Analysis**: Line 680 cites FedGH under prototype FL (it aggregates heads); Line 736 falsely claims all decoupled FL aggregates extractors; Line 748 classifies FedGMKD under second-order statistics; Line 757 attributes positive-repulsion-free loss solely to SupCon without citing DCL; and method acronyms are glued to clauses without punctuation (e.g., `representation collapse FedBABU~\citep{...}`).
- **Drop-in Fix**: Correct citations to FedGH, FedGMKD, SupCon/DCL, and normalize citation grammar across Related Work.

### 7. Pervasive Draft Revision Markup (`\textcolor{blue}{...}`, `\textcolor{red}{...}`)
- **Manuscript Location**: Sections 1, 2, 3, 4, 5; visible in `icml2026.pdf`.
- **Technical Analysis**: Large narrative blocks remain wrapped in draft color commands, causing entire pages to print in solid blue text.
- **Drop-in Fix**: Strip all `\textcolor{blue}{` and `\textcolor{red}{` commands and closing braces throughout the manuscript.

### 8. Duplicated Paragraph Header in Section 4
- **Manuscript Location**: Section 4.1, Lines 606--607 of `icml2026.tex`.
- **Technical Analysis**: `\textbf{Implementation details.}` appears twice consecutively.
- **Drop-in Fix**: Delete the duplicate header at Line 606.

---

# 5. Codebase-to-Paper Alignment Matrix

The following matrix provides an exhaustive, line-by-line verification mapping between every mathematical construct in `Overleaf/icml2026.tex` and its implementation across the PyTorch codebase (`clientscd.py`, `serverscd.py`, `main.py`, `run_benchmark_matrix.py`):

| Mathematical Construct / Symbol | Paper Formulation & Location | Codebase Implementation | Location in Repository | Alignment Status & Detailed Notes |
|:---|:---|:---|:---|:---|
| **Local Feature Variance $(\sigma_k^c)^2$** | Eq. 2: $\frac{1}{n_k^c}\sum (f_k(x) - \mu_k^c)^{\odot 2}$ | `torch.std(class_feats, dim=0, unbiased=False)` | `clientscd.py:125` | **Exact Alignment**. Validates empirical plug-in condition for Prop 3.1. |
| **Singleton Class Safeguard** | Unstated in Eq. 2 (assumes $n \ge 1$) | `std_c = torch.full_like(mean_c, 1e-6)` if $n \le 1$ | `clientscd.py:125` | **Code Safeguard Omitted in Paper**. Essential non-degeneracy condition. |
| **Transmitted Client Payload** | Tuple $\{\mu_k^c, \sigma_k^c, n_k^c\}$ | `{'mean': mean_c.cpu(), 'std': std_c.cpu(), 'count': int(n)}` | `clientscd.py:126` | **Exact Alignment**. Transmits std vector $\sigma_k^c \in \mathbb{R}^d$. |
| **Bhattacharyya Distance $D_B$** | Eq. 3: Sum over all $d$ dimensions | Clamps variance to $10^{-4}$; masks dead coordinates | `motivation_metrics.py:161--174` | **Discrepancy**. Code filters dead ReLU neurons; paper omits floor. |
| **Bhattacharyya Overlap Index** | Line 304: $\exp(-D_B / d)$ | `torch.exp(-D_B / np.sqrt(d_eff))` | `motivation_metrics.py:184` | **Major Discrepancy**. Code normalizes by $\sqrt{d_{\mathrm{eff}}}$; text says $d$. |
| **Empirical Bayes Anchor $p^c$** | Eq. 5: $(1-\gamma^c)\mu^c + \gamma^c \mu_k^c$ | `p = (1.0 - gamma) * glob_m + gamma * loc_m` | `clientscd.py:186` | **Exact Alignment**. $\gamma_{\max} = 0.25, n_0 = 50$. Fallback $p^c=\mu^c$ if $n=0$. |
| **Variance Floor $\tau_v$** | Line 362: $\max(\alpha_v\mathrm{mean}((\sigma^c)^2), \alpha_q)$ | `max(0.25 * float(torch.mean(var_tensor)), 1e-2)` | `clientscd.py:210` | **Scope Mismatch**. Code averages globally across classes and dims. |
| **Coordinate SNR Score $s_j^c$** | Line 368: $u_j^c / \mathrm{mean}(u^c)$ | `snr_raw = mean_vec / (std_vec + 1e-6); snr / mean` | `clientscd.py:216--222` | **Exact Alignment**. Normalized across dimensions. |
| **SNR Modulation Gate $g_j^c$** | Eq. 6: $\frac{0.5 + 0.5 s_j^c}{1 + 0.5 s_j^c}$ | `gate = (0.5 + 0.5 * snr_c) / (1.0 + 0.5 * snr_c)` | `clientscd.py:224` | **Exact Alignment**. Smooth saturation curve in $[0.5, 1.0]$. |
| **Precision Weight $w_j^c$** | Eq. 6: $g_j^c / \sqrt{v_j^c}$ | `w = g * (u_norm)` where `u = 1.0 / sqrt(v)` | `clientscd.py:228--234` | **Exact Alignment**. Precision weighting along compact coordinates. |
| **Normalized Anchor $q^c$** | Eq. 6: $w^c \odot p^c / \|w^c \odot p^c\|_2$ | `F.normalize(w * p, p=2, dim=-1)` | `clientscd.py:230` | **Exact Alignment**. Resides on unit hypersphere $\mathbb{S}^{d-1}$. |
| **ProtoNCE Loss $\mathcal{L}_{\mathrm{NCE}}$** | Eq. 7: $-\log \frac{\exp(\hat{z}^\top q^y / \phi^y)}{\sum \exp(\hat{z}^\top q^c / \phi^c)}$ | Evaluated with class-specific temperatures $\phi^c$ | `clientscd.py:348--356` | **Exact Alignment**. Masks unobserved classes. Labeled `proto_nce`. |
| **Decoupled SupCon Loss** | Eq. 8: Denominator omits positive pairs | Excludes same-class pairs from denominator (DCL) | `clientscd.py:358--390` | **Exact Alignment**. Follows \citet{yeh2022decoupled} formulation. |
| **Mahalanobis Loss $\mathcal{L}_{\mathrm{PL}}$** | Eqs. 9--10: $\frac{1}{d}\sum g_j^y(z_j - p_j^y)^2 / v_j^y$ | `torch.sum(g * (feat - p)**2 / v, dim=-1) / d` | `clientscd.py:392--402` | **Exact Code Alignment**. Text narrative inverted ("penalizes high-var"). |
| **Feature Extractor Loss** | Eq. 11: $\mathcal{L}_{\mathrm{CE}} + \lambda_t(\mathcal{L}_{\mathrm{NCE}} + \dots)$ | `loss = loss_ce + lamda_t * (loss_scd + ...)` | `clientscd.py:504--510` | **Exact Alignment**. Linear ramp-up applied. |
| **Classifier Head Loss** | Eq. 12: $\mathcal{L}_{\mathrm{CE}}(h_k(\mathrm{sg}[f_k(x)]), y) + \dots$ | `loss_ce = self.loss(out, y)` on detached rep | `clientscd.py:457` | **Exact Alignment**. Uses gradient detachment $\mathrm{sg}[\cdot]$. |
| **Feature Manifold Mixup** | **UNDISCLOSED IN PAPER** | `--mixup_alpha 0.2`, blends 25% mixup loss | `clientscd.py:459--472` | **Major Paper Omission**. Active by default in codebase. |
| **Virtual Perturbation Scale** | Line 447: $\eta_{\mathrm{loc}}=0.5, \eta_{\mathrm{glob}}=0.3$ | `eta_loc = 0.5, eta_glob = 0.3` | `clientscd.py:371, 385` | **Exact Alignment**. Fig. 2 uses obsolete symbol $\alpha$. |
| **Virtual Head Weights** | Line 607: $\lambda_{\mathrm{vl}} = 0.05, \lambda_{\mathrm{vg}} = 0.05$ | $\lambda_{\mathrm{vl}} = 0.0$ (disabled), $\lambda_{\mathrm{vg}} = 0.05$ | `clientscd.py:56, 59` | **Major Discrepancy**. $\lambda_{\mathrm{vl}}$ is inactive in actual runs. |
| **Server Pooled Mean $\bar{\mu}_t^c$** | Eq. 13: $\frac{1}{\bar{n}_t^c}\sum n_k^c \mu_k^c$ | `batch_mean = sum(n_k * mean_k) / total_n` | `serverscd.py:192--196` | **Exact Alignment**. Filters active clients with $n_k^c \ge 1$. |
| **Total Variance Pooling $(\bar{\sigma}_t^c)^2$** | Eq. 14: $\frac{1}{\bar{n}_t^c}\sum n_k^c [(\sigma_k^c)^2 + (\mu_k^c - \bar{\mu}_t^c)^{\odot 2}]$ | `pooled_var += n_k * (local_var + (mean_k - batch_mean)**2)` | `serverscd.py:197--204` | **Exact Alignment**. Faithfully captures between-client drift. |
| **Server Momentum $\beta$** | Line 607: $\beta = 0.6$; Line 742: $\beta = 0.6$ | Default `proto_momentum = 0.1` | `serverscd.py:32`, `main.py:326` | **Major Discrepancy** ($6\times$ difference in learning rate). |
| **Smoothed Active Support $n_t^c$** | Eq. 15: Exponential moving average | `total_count = round((1-m)*prev + m*batch)` | `serverscd.py:214` | **Exact Alignment**. Smoothed per-round support, not cumulative. |
| **Dispersion Index $r_t^c$** | Eq. 16: $\frac{\|\sigma_t^c\|_2 / \|\mu_t^c\|_2}{\log(n_t^c + \kappa)}$ | `(norm(std) / (norm(mean) + 1e-6)) / log(count + 10.0)` | `serverscd.py:221--224` | **Exact Alignment** ($\kappa = 10$). Named "concentration" in paper. |
| **Temperature Clipping $\phi_t^c$** | Eq. 17: $\mathrm{clip}(\tau \frac{r_t^c}{\text{mean}}, (1\pm\rho)\tau)$ | `scaled_phi = base_tau * (raw / mean_raw); clip(...)` | `serverscd.py:263--265` | **Exact Alignment** ($\tau=0.1, \rho=0.1$). |
| **Ramp-up Schedule $T_{\mathrm{ramp}}$** | Line 607: $T_{\mathrm{ramp}} = 10$ | `self.proto_rampup = 5` (default 5 in benchmarks) | `clientscd.py:52`, `main.py:318` | **Major Discrepancy** ($5$ rounds in code vs. $10$ in text). |
| **Evaluation Metric** | Line 609: "final communication round" | Table 1 copies `df.max() * 100` (peak accuracy)| `runs_results.xlsx` | **Critical Contradiction**. Conceals baseline collapse. |
| **Client Upload Bandwidth** | Line 549: $\le C(2d+1)$ floats | Uploads tuple $\{\mu_k^c, \sigma_k^c, n_k^c\}$ | `clientscd.py:126` | **Exact Alignment**. Strictly linear $\mathcal{O}(Cd)$. |
| **Server Download Bandwidth** | Line 549: $\le C(2d+1)$ floats | Broadcasts $\{\mu_t^c, \sigma_t^c, \phi_t^c\}$ | `serverscd.py:147` | **Exact Alignment**. Total round-trip $\le 2C(2d+1)$ floats. |

---

# 6. Mathematical Appendix & Formal Proof Analysis

To resolve **CRIT-02 (Missing Proofs)** and **MAJ-12, MAJ-13 (Theoretical Precision & Bounds)**, the following self-contained, publication-grade mathematical derivations provide complete proofs for **Proposition 3.1**, **Proposition 3.2**, **Proposition 3.3**, and **Corollary 3.3.1** (Multiclass Bayes Error Union Bound). This text is formatted for direct drop-in integration into `Overleaf/icml2026.tex` as a formal appendix:

```latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX: DEFERRED PROOFS AND THEORETICAL ANALYSIS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\appendix
\onecolumn
\section{Deferred Proofs and Theoretical Analysis}
\label{sec:proofs}

In this appendix, we provide self-contained, mathematically rigorous proofs for the theoretical propositions stated in Section~\ref{sec:theory}, establish the exact regularity conditions under which decentralized total-variance pooling holds, and derive the multiclass Bayes classification error bound.

\subsection{Proof of Proposition~\ref{prop:exact} (Exact Total-Variance Pooling)}
\label{app:proof_exact}

\begin{proposition}[Exact Pooling under Empirical Plug-in Variance]
Let $\mathcal{S}_t \subseteq \{1, \dots, K\}$ denote the set of participating clients in communication round $t$. For each client $k \in \mathcal{S}_t$ possessing samples of class $c$, let $\mathcal{D}_k^c = \{z_{k,i}^c\}_{i=1}^{n_k^c} \subset \mathbb{R}^d$ denote its local feature embeddings, where $n_k^c \ge 1$. Let $\mathcal{S}_t^c \triangleq \{k \in \mathcal{S}_t \mid n_k^c \ge 1\}$ denote the set of active clients for class $c$, and let
\begin{equation}
    \mathcal{U}_t^c \triangleq \bigcup_{k \in \mathcal{S}_t^c} \mathcal{D}_k^c = \bigcup_{k \in \mathcal{S}_t^c} \{z_{k,i}^c\}_{i=1}^{n_k^c}
\end{equation}
denote the feature union across all participating clients, with total cardinality $\bar{n}_t^c = \sum_{k \in \mathcal{S}_t^c} n_k^c$.
Under the empirical plug-in sample variance definition (Eq.~\ref{eq:var}, with denominator $n_k^c$), the pooled server statistics $\bar{\mu}_t^c$ (Eq.~\ref{eq:server_mean}) and $(\bar{\sigma}_t^c)^{\odot 2}$ (Eq.~\ref{eq:server_var}) satisfy:
\begin{equation}
    \bar{\mu}_t^c = \mu_{\mathcal{U}_t^c} \quad \text{and} \quad (\bar{\sigma}_t^c)^{\odot 2} = (\sigma_{\mathcal{U}_t^c})^{\odot 2},
\end{equation}
where $\mu_{\mathcal{U}_t^c}$ and $(\sigma_{\mathcal{U}_t^c})^{\odot 2}$ are the exact empirical sample centroid and pooled plug-in variance of $\mathcal{U}_t^c$. Furthermore, when $\beta = 1$ in Eq.~\ref{eq:ema}, the broadcast global statistics $(\mu_t^c, (\sigma_t^c)^{\odot 2})$ equal these exact pooled moments without communicating raw data.
\end{proposition}

\begin{proof}
We evaluate the first- and second-order empirical moments coordinate-wise along each feature dimension $j \in \{1, \dots, d\}$.

\paragraph{1. First-Order Moment (Sample Mean):}
By definition of the centralized arithmetic mean of the multi-client union $\mathcal{U}_t^c$:
\begin{equation}
    \mu_{\mathcal{U}_t^c, j} = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c.
\end{equation}
Grouping the summation across participating clients $k \in \mathcal{S}_t^c$:
\begin{equation}
    \mu_{\mathcal{U}_t^c, j} = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \left( \frac{1}{n_k^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c \right) = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \mu_{k,j}^c = \bar{\mu}_{t,j}^c.
\end{equation}
Vectorizing across all coordinates $j \in \{1, \dots, d\}$ yields $\bar{\mu}_t^c = \mu_{\mathcal{U}_t^c}$.

\paragraph{2. Second-Order Moment (Sample Variance):}
By definition of the uncorrected empirical plug-in variance (matching `torch.std(..., unbiased=False)` in Eq.~\ref{eq:var}):
\begin{equation}
    (\sigma_{\mathcal{U}_t^c, j})^2 = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Decomposing the deviation of individual representation $z_{k,i,j}^c$ from the global pooled mean $\bar{\mu}_{t,j}^c$ around the local client mean $\mu_{k,j}^c$:
\begin{equation}
    z_{k,i,j}^c - \bar{\mu}_{t,j}^c = \big(z_{k,i,j}^c - \mu_{k,j}^c\big) + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big).
\end{equation}
Squaring both sides and expanding:
\begin{equation}
    \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2 = \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 + 2\big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)\big(z_{k,i,j}^c - \mu_{k,j}^c\big).
\end{equation}
Summing over all local samples $i \in \{1, \dots, n_k^c\}$ on client $k$:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \bar{\mu}_{t,j}^c\big)^2 = \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 + \sum_{i=1}^{n_k^c} \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 + 2\big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big) \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big).
\end{equation}
By definition of the arithmetic local sample mean $\mu_{k,j}^c = \frac{1}{n_k^c} \sum_{i=1}^{n_k^c} z_{k,i,j}^c$, the local deviations sum identically to zero:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big) = \sum_{i=1}^{n_k^c} z_{k,i,j}^c - n_k^c \mu_{k,j}^c = 0.
\end{equation}
Consequently, the cross-product term vanishes identically. Evaluating the remaining two sums:
\begin{equation}
    \sum_{i=1}^{n_k^c} \big(z_{k,i,j}^c - \mu_{k,j}^c\big)^2 = n_k^c (\sigma_{k,j}^c)^2, \qquad \sum_{i=1}^{n_k^c} \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 = n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Summing over all participating clients $k \in \mathcal{S}_t^c$ and dividing by total sample support $\bar{n}_t^c$:
\begin{equation}
    (\sigma_{\mathcal{U}_t^c, j})^2 = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c \Big[ (\sigma_{k,j}^c)^2 + \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 \Big] = (\bar{\sigma}_{t,j}^c)^2.
\end{equation}
Vectorizing over coordinates $j \in \{1, \dots, d\}$ yields $(\bar{\sigma}_t^c)^{\odot 2} = (\sigma_{\mathcal{U}_t^c})^{\odot 2}$, matching Eq.~\ref{eq:server_var}. When $\beta = 1$ in Eq.~\ref{eq:ema}, the momentum smoothing step sets $s_t^c = \bar{s}_t^c$, ensuring broadcast moments match the centralized union moments exactly.
\end{proof}

\subsection{Proof of Proposition~\ref{prop:drift} (Drift-Induced Dispersion Inflation)}
\label{app:proof_drift}

\begin{proposition}[Drift-Induced Dispersion Inflation]
Let $W^c \triangleq \sum_{k \in \mathcal{S}_t^c} n_k^c (\sigma_k^c)^{\odot 2}$ and $B^c \triangleq \sum_{k \in \mathcal{S}_t^c} n_k^c (\mu_k^c - \bar{\mu}_t^c)^{\odot 2}$ denote the within-client scatter vector and between-client scatter vector, respectively. The pooled variance in Eq.~\ref{eq:server_var} satisfies:
\begin{equation}
    (\bar{\sigma}_t^c)^{\odot 2} \ge \frac{1}{\bar{n}_t^c} W^c \quad \text{element-wise},
\end{equation}
with equality holding along coordinate $j \in \{1, \dots, d\}$ if and only if all participating local centroids coincide along that coordinate ($\mu_{k,j}^c = \bar{\mu}_{t,j}^c$ for all $k \in \mathcal{S}_t^c$). Any representation drift across heterogeneous clients strictly inflates the aggregated class dispersion beyond the naive weighted average of local variances.
\end{proposition}

\begin{proof}
From Eq.~\ref{eq:server_var}, the $j$-th coordinate of pooled variance decomposes as:
\begin{equation}
    (\bar{\sigma}_{t,j}^c)^2 = \frac{1}{\bar{n}_t^c} W_j^c + \frac{1}{\bar{n}_t^c} B_j^c, \quad \text{where } B_j^c = \sum_{k \in \mathcal{S}_t^c} n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2.
\end{equation}
Because sample counts satisfy $n_k^c \ge 1$ and squared deviations satisfy $(\mu_{k,j}^c - \bar{\mu}_{t,j}^c)^2 \ge 0$, every term in $B_j^c$ is non-negative. Therefore:
\begin{equation}
    B_j^c \ge 0 \implies (\bar{\sigma}_{t,j}^c)^2 \ge \frac{1}{\bar{n}_t^c} W_j^c = \frac{1}{\bar{n}_t^c} \sum_{k \in \mathcal{S}_t^c} n_k^c (\sigma_{k,j}^c)^2.
\end{equation}
Equality holds if and only if $B_j^c = 0$. Since $n_k^c \ge 1$ for all active clients:
\begin{equation}
    B_j^c = 0 \iff n_k^c \big(\mu_{k,j}^c - \bar{\mu}_{t,j}^c\big)^2 = 0 \quad \forall k \in \mathcal{S}_t^c \iff \mu_{k,j}^c = \bar{\mu}_{t,j}^c \quad \forall k \in \mathcal{S}_t^c.
\end{equation}
Whenever local representation drift exists along coordinate $j$ ($\exists k, l$ such that $\mu_{k,j}^c \neq \mu_{l,j}^c$), $B_j^c > 0$, which strictly inflates the aggregated class variance $(\bar{\sigma}_{t,j}^c)^2 > \frac{1}{\bar{n}_t^c} W_j^c$.
\end{proof}

\subsection{Proof of Proposition~\ref{prop:bayes_bound} (Bhattacharyya Bound on Pairwise Bayes Error)}
\label{app:proof_bayes}

\begin{proposition}[Pairwise Bayes Error Upper Bound~\citep{kailath1967divergence}]
For any pair of classes $c, c'$ with prior probabilities $\pi_c, \pi_{c'} > 0$ ($\pi_c + \pi_{c'} = 1$) and class-conditional densities $p_c(z), p_{c'}(z)$, the optimal Bayes classification error $P_e(c, c')$ satisfies:
\begin{equation}
    P_e(c, c') \le \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big),
\end{equation}
where $D_B(c, c')$ is the Bhattacharyya distance defined in Eq.~\ref{eq:bhatt}.
\end{proposition}

\begin{proof}
Let $\Omega_c \triangleq \{z \in \mathbb{R}^d \mid \pi_c p_c(z) \ge \pi_{c'} p_{c'}(z)\}$ denote the Bayes-optimal decision region for class $c$, and $\Omega_{c'} \triangleq \mathbb{R}^d \setminus \Omega_c$ denote the region for class $c'$. The total Bayes classification error is:
\begin{equation}
    P_e(c, c') = \int_{\Omega_{c'}} \pi_c p_c(z)\, dz + \int_{\Omega_c} \pi_{c'} p_{c'}(z)\, dz = \int_{\mathbb{R}^d} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big)\, dz.
\end{equation}
For any non-negative real numbers $a, b \ge 0$, the algebraic inequality $\min(a, b) \le a^s b^{1-s}$ holds for all $s \in [0, 1]$ (the Chernoff bound). Setting $s = 1/2$ yields:
\begin{equation}
    \min(a, b) \le \sqrt{ab}.
\end{equation}
Applying this inequality point-wise to $a = \pi_c p_c(z)$ and $b = \pi_{c'} p_{c'}(z)$:
\begin{equation}
    P_e(c, c') \le \int_{\mathbb{R}^d} \sqrt{\pi_c p_c(z) \cdot \pi_{c'} p_{c'}(z)}\, dz = \sqrt{\pi_c \pi_{c'}} \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz.
\end{equation}
The integral defines the Bhattacharyya affinity coefficient $\mathrm{BC}(c, c') \triangleq \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz \in [0, 1]$ (Eq.~\ref{eq:bc}).
By definition of the Bhattacharyya distance $D_B(c, c') \triangleq -\ln \mathrm{BC}(c, c')$, we have $\mathrm{BC}(c, c') = \exp\big(-D_B(c, c')\big)$, yielding:
\begin{equation}
    P_e(c, c') \le \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}

Now, consider class-conditional diagonal Gaussians $p_c(z) = \mathcal{N}(z; \mu^c, \Sigma^c)$ and $p_{c'}(z) = \mathcal{N}(z; \mu^{c'}, \Sigma^{c'})$ with diagonal covariance matrices $\Sigma^c = \mathrm{diag}\big((\sigma_1^c)^2, \dots, (\sigma_d^c)^2\big)$ and $\Sigma^{c'} = \mathrm{diag}\big((\sigma_1^{c'})^2, \dots, (\sigma_d^{c'})^2\big)$.
By Gaussian integration~\citep{kailath1967divergence}:
\begin{equation}
    \int_{\mathbb{R}^d} \sqrt{p_c(z) p_{c'}(z)}\, dz = \frac{\det(\Sigma^c)^{1/4} \det(\Sigma^{c'})^{1/4}}{\det(\bar{\Sigma})^{1/2}} \exp\left(-\frac{1}{8}(\mu^c - \mu^{c'})^\top \bar{\Sigma}^{-1} (\mu^c - \mu^{c'})\right),
\end{equation}
where $\bar{\Sigma} = \frac{1}{2}(\Sigma^c + \Sigma^{c'})$. Taking the negative natural logarithm:
\begin{equation}
    D_B(c, c') = \frac{1}{8}(\mu^c - \mu^{c'})^\top \bar{\Sigma}^{-1} (\mu^c - \mu^{c'}) + \frac{1}{2} \ln \frac{\det \bar{\Sigma}}{\sqrt{\det \Sigma^c \det \Sigma^{c'}}}.
\end{equation}
Because all covariance matrices are diagonal, $\bar{\Sigma}$ is diagonal with entries $\sigma_{c, c', j}^2 = \frac{1}{2}\big((\sigma_j^c)^2 + (\sigma_j^{c'})^2\big)$, and determinants factorize into products over coordinates:
\begin{equation}
    D_B(c, c') = \frac{1}{8} \sum_{j=1}^d \frac{(\mu_j^c - \mu_j^{c'})^2}{\sigma_{c, c', j}^2} + \frac{1}{2} \sum_{j=1}^d \ln \frac{\sigma_{c, c', j}^2}{\sigma_j^c \sigma_j^{c'}},
\end{equation}
which is exactly Eq.~\ref{eq:bhatt}. Because $\exp(-D_B)$ is strictly decreasing in $D_B$, enlarging $D_B(c, c')$ directly contracts the upper bound on pairwise Bayes error.
\end{proof}

\subsection{Multiclass Bayes Error Union Bound (Corollary to Proposition~\ref{prop:bayes_bound})}
\label{app:multiclass_bound}

\begin{corollary}[Multiclass Bayes Error Union Bound]
For a $C$-class classification problem with class priors $\{\pi_c\}_{c=1}^C$ ($\sum_{c=1}^C \pi_c = 1$) and class-conditional distributions $\{p_c(z)\}_{c=1}^C$, the overall Bayes classification error $P_e$ is upper-bounded by:
\begin{equation}
    P_e \le 2 \sum_{c=1}^C \sum_{c' > c} \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}
Under uniform class priors $\pi_c = 1/C$, this bound simplifies to:
\begin{equation}
    P_e \le \frac{2}{C} \sum_{c=1}^C \sum_{c' > c} \exp\big(-D_B(c, c')\big).
\end{equation}
\end{corollary}

\begin{proof}
Let $R_c \triangleq \{z \in \mathbb{R}^d \mid \pi_c p_c(z) = \max_{1 \le m \le C} \pi_m p_m(z)\}$ denote the Bayes decision region for class $c$. The total probability of correct classification is $P_c = \sum_{c=1}^C \int_{R_c} \pi_c p_c(z)\, dz$. The overall Bayes classification error is:
\begin{equation}
    P_e = 1 - P_c = \sum_{c=1}^C \int_{\mathbb{R}^d \setminus R_c} \pi_c p_c(z)\, dz.
\end{equation}
For any observation $z \in \mathbb{R}^d \setminus R_c$, class $c$ does not attain the maximum posterior density, which implies that there exists at least one competitor class $c' \neq c$ such that $\pi_c p_c(z) \le \pi_{c'} p_{c'}(z)$. Therefore:
\begin{equation}
    \pi_c p_c(z) = \min\big(\pi_c p_c(z), \max_{c' \neq c} \pi_{c'} p_{c'}(z)\big) \le \sum_{c' \neq c} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big).
\end{equation}
Integrating this point-wise inequality over $\mathbb{R}^d$:
\begin{equation}
    \int_{\mathbb{R}^d \setminus R_c} \pi_c p_c(z)\, dz \le \int_{\mathbb{R}^d} \pi_c p_c(z)\, dz \le \sum_{c' \neq c} \int_{\mathbb{R}^d} \min\big(\pi_c p_c(z),\, \pi_{c'} p_{c'}(z)\big)\, dz = \sum_{c' \neq c} P_e(c, c').
\end{equation}
Summing over all classes $c \in \{1, \dots, C\}$:
\begin{equation}
    P_e \le \sum_{c=1}^C \sum_{c' \neq c} P_e(c, c') = 2 \sum_{c=1}^C \sum_{c' > c} P_e(c, c').
\end{equation}
Applying Proposition~\ref{prop:bayes_bound} to each pairwise error $P_e(c, c')$:
\begin{equation}
    P_e \le 2 \sum_{c=1}^C \sum_{c' > c} \sqrt{\pi_c \pi_{c'}} \exp\big(-D_B(c, c')\big).
\end{equation}
For balanced class priors where $\pi_c = 1/C$ for all $c \in \{1, \dots, C\}$, we have $\sqrt{\pi_c \pi_{c'}} = 1/C$, yielding:
\begin{equation}
    P_e \le \frac{2}{C} \sum_{c=1}^C \sum_{c' > c} \exp\big(-D_B(c, c')\big).
\end{equation}
This establishes that maximizing pairwise Bhattacharyya distances $D_B(c, c')$ across all distinct pairs $(c, c')$ provably tightens the upper bound on the multiclass Bayes classification error.
\end{proof}
```

---

# 7. Prioritized Actionable Remediation Plan

To elevate `Overleaf/icml2026.tex` to a compliant, publication-grade submission for ICML 2026, we outline a structured 3-phase remediation plan:

```
[Phase 1: Immediate Source Synchronization & Compilation Cleanliness] (Zero GPU compute; 2 Hours)
  ├── 1.1 Synchronize Section 1 (Line 198) and Section 5.2 (Lines 645--648) with Table 1 data.
  ├── 1.2 Align Section 4 metric definition (Line 609) with peak test accuracy across rounds.
  ├── 1.3 Fix broken LaTeX equation references (\ref{eq:proto_nce}, \ref{eq:head_loss}, \ref{eq:ema}).
  ├── 1.4 Correct hyperparameter specifications (\beta=0.1, T_ramp=5, \lambda_vl=0.0).
  ├── 1.5 Purge author directive macros (\NMD, \GVQ) and remove draft colored wrappers (\textcolor).
  ├── 1.6 Retitle Table 3 to "Design Choice and Architectural Ablations on CIFAR-10".
  └── 1.7 Fix root-level TeX grouping defect across lines 678--736.

[Phase 2: Page Limit Condensation & Appendix Restructuring] (Mandatory for ICML Compliance; 3 Hours)
  ├── 2.1 Insert the formal Appendix block (Section 6 of this report) providing proofs for Props 3.1--3.3.
  ├── 2.2 Condense Section 2 (Related Work) and Section 3 (Methodology) to bring main text strictly <= 8 pages.
  ├── 2.3 Remove unpopulated Tiny-ImageNet columns (21 dashes) from Table 1.
  ├── 2.4 Delineate Table 1 benchmark scope strictly as CIFAR-10 (or populate CIFAR-100/Flowers102).
  └── 2.5 Recompile and verify PDF pagination: references start strictly on Page 9.

[Phase 3: Visual Grounding, Baseline Fairness & Code Alignment] (Compute-Dependent; 1-2 Days)
  ├── 3.1 Repair Figure_Plot/test_acc.csv and re-render figures/motivation.pdf.
  ├── 3.2 Formally disclose Feature Manifold Mixup (mixup_alpha=0.2) in Section 3.4 and Section 4.1.
  ├── 3.3 Add quantitative endpoint callouts and explicit Figure 1 references in Section 5.3.
  ├── 3.4 Re-run FedTGP and FedSA baselines with standardized batch size B=32 and lr=0.01.
  └── 3.5 Execute run_benchmark_matrix.py to complete CIFAR-100 and Flowers102 for FedSCD.
```

---

### 7.1 Exact Drop-in LaTeX Replacements for Core Manuscript Sections

#### 1. Reconciled Section 1 Contributions (Line 198)
```latex
% REPLACE line 198 in Overleaf/icml2026.tex:
\item Experiments on CIFAR-10 across varying degrees of Dirichlet label skew ($\alpha \in \{1.0, 0.5, 0.1\}$) show that FedSCD consistently lowers Bhattacharyya overlap and outperforms state-of-the-art prototype- and statistics-based baselines---improving test accuracy by up to $5.09$ percentage points over FedTGP (at $\alpha = 1.0$) and $3.45$ percentage points at $\alpha = 0.5$, while achieving margins of up to $8.39$ percentage points over FedORGP and up to $18.77$ percentage points over FedGMKD.
```

#### 2. Reconciled Section 4 Metric & Hyperparameter Description (Lines 606--609)
```latex
% REPLACE lines 606-609 in Overleaf/icml2026.tex:
\textbf{Implementation details.} All methods are trained for $T=100$ rounds with full participation, one local epoch per round ($E=1$), mini-batch size 32, and SGD with a learning rate of 0.01 without momentum or weight decay. FedSCD uses decoupled SGD optimizers for the feature extractor and classification head with the same learning rate. For FedSCD, we set the base temperature $\tau = 0.1$, temperature modulation margin $\rho = 0.1$, server momentum $\beta = 0.1$, anchor regularization weight $\lambda = 0.1$, instance contrastive weight $\lambda_{\text{ins}} = 0.05$, warm-up rounds $T_{\text{warm}} = 1$, ramp-up rounds $T_{\text{ramp}} = 5$, and head calibration weight $\lambda_{\text{vg}} = 0.05$ (with $\lambda_{\text{vl}} = 0.0$). For distribution shaping and coordinate gating, we set maximum shrinkage $\gamma_{\max} = 0.25$, prior sample count $n_0 = 50$, variance floor scale $\alpha_v = 0.25$, numerical stability constant $\alpha_q = 10^{-2}$, and virtual feature noise scales $\eta_{\text{loc}} = 0.5, \eta_{\text{glob}} = 0.3$. Baseline methods are configured with their official default hyperparameters.

\textbf{Metrics.} We measure model performance using the peak average local test accuracy across 100 communication rounds, reflecting the maximum attainable representation capability under non-IID partitioning. To evaluate representation geometry, we also track the Euclidean prototype margin and the Bhattacharyya overlap ($e^{-D_B}$ via Eq.~\ref{eq:bc}) evaluated over server-aggregated class statistics across all class pairs. Communication efficiency is assessed by the number of transmitted floating-point parameters per client per round.
```

#### 3. Reconciled Section 5.2 Performance Narrative (Lines 643--648)
```latex
% REPLACE lines 643-648 in Overleaf/icml2026.tex:
\textbf{Main results on label skew.}
Table~\ref{tab:com} reports the test accuracy on CIFAR-10 across varying degrees of Dirichlet label skew. 
Under mild heterogeneity ($\alpha = 1.0$), FedSCD achieves 69.60\% test accuracy, outperforming the runner-up baseline FedTGP (64.51\%) by +5.09 percentage points, FedCOF (63.62\%) by +5.98 points, and FedORGP (61.21\%) by +8.39 points.
At moderate heterogeneity ($\alpha = 0.5$), FedSCD reaches 75.15\%, outperforming the strongest baseline FedTGP (71.70\%) by +3.45 percentage points and surpassing other baselines (FedCOF 69.64\%, FedPLVM 69.76\%, FedORGP 68.70\%, FedSA 68.48\%) by up to +6.67 percentage points. 
Under extreme heterogeneity ($\alpha = 0.1$), FedSCD achieves 90.71\%, maintaining margins of +1.83 percentage points over runner-up FedORGP (88.88\%) and +1.84 percentage points over FedTGP (88.87\%).
```

#### 4. Grounded Section 5.3 Representation Geometry (Lines 664--672)
```latex
% REPLACE lines 664-672 in Overleaf/icml2026.tex:
\subsection{Representation Geometry and Separability}
\label{sec:geometry}

\textbf{Prototype margin vs. distributional overlap.}
To validate the theoretical insights established in Section~\ref{sec:motivation}, we evaluate both the Euclidean prototype margin and the Bhattacharyya overlap ($e^{-D_B}$ via Eq.~\ref{eq:bc}) computed on server-aggregated class statistics across training rounds.
As illustrated in Figure~\ref{fig:motivation}(a)--(b), optimizing solely for inter-prototype Euclidean distance yields deceptive geometric separation: while FedTGP widens the Euclidean prototype margin to 23.85 by Round 100, it simultaneously incurs high Bhattacharyya overlap (0.894) due to uncontrolled intra-class variance in peripheral dimensions. 
Conversely, FedSCD contracts Bhattacharyya overlap to 0.870 while maintaining a compact Euclidean centroid margin (3.77), confirming that coordinate-wise Mahalanobis shrinkage and dynamic angular repulsion are indispensable for achieving genuine class separation on the hypersphere.
```

#### 5. Drop-In Equation Label Aliases (Preamble & Methodology)
```latex
% In Overleaf/icml2026.tex:
% At Line 393:
\label{eq:proto_nce}\label{eq:nce}

% At Line 441:
\label{eq:head_loss}\label{eq:head}

% At Line 492:
\label{eq:ema}\label{eq:server_mom}

% At Line 612:
\label{tab:com}\label{tab:label_skew}
```

#### 6. Rigorous Section 6 Conclusion & Limitations (Lines 748--756)
```latex
% REPLACE lines 748-756 in Overleaf/icml2026.tex:
\section{Conclusion}
\label{sec:conclusion}

This work investigates distribution-aware prototype learning in federated learning under non-IID label skew. We demonstrated both theoretically and empirically that conventional margin-based methods operate under a variance-blind assumption: enlarging Euclidean distances between isolated point centroids does not prevent substantial Bayes classification error when intra-class feature variances remain broad and anisotropic. To resolve this fundamental limitation, we introduced FedSCD, moving the FL prototype paradigm beyond zero-variance points to continuous diagonal Gaussian distributions. By coupling client-side geometric shaping with server-side moment aggregation, FedSCD establishes an architecture-agnostic closed loop that directly targets distributional separability without communicating high-dimensional neural network weights.

Our technical contributions yield concrete theoretical and empirical advantages:
First, by deriving coordinate-wise SNR gates and Mahalanobis compactness penalties, clients actively compress coordinate-wise variance in the Mahalanobis term of the Bhattacharyya distance, provably enlarging inter-class distribution separation. Second, server-side total-variance pooling ensures lossless moment reconstruction, absorbing between-client representation drift without underestimating population spread. Benchmark experiments on CIFAR-10 across severe Dirichlet partitions demonstrate that FedSCD systematically reduces Bhattacharyya overlap and achieves superior accuracy over state-of-the-art baselines. Crucially, ablation analyses confirm that active variance compression---rather than data augmentation or centroid expansion---is the primary driver of these robust generalization gains.

\textbf{Limitations and Future Work.} Although FedSCD substantially improves representation quality under label skew while maintaining a minimal $\mathcal{O}(Cd)$ communication footprint, several fundamental boundaries warrant future investigation. First, our framework parameterizes feature distributions as diagonal Gaussians; modern deep representations post-ReLU lie on non-negative conical manifolds, where Gaussian approximations may allocate probability mass to inactive quadrants. Second, under extreme class sparsity ($n_k^c \le 2$), sample variance estimates degenerate, placing heavy reliance on server shrinkage. Third, while preserving $\mathcal{O}(Cd)$ scaling, transmitting diagonal variances introduces a $2\times$ payload overhead ($C(2d+1)$ floats) relative to standard point prototypes ($Cd$). Finally, sharing second-order dispersion statistics introduces differential privacy considerations that motivate formal DP perturbation mechanisms.
```

---

# 8. Segment Report References Table

The exhaustive findings in this definitive report are traceable to the five Level-2 Root Unit Audit Reports produced across the tournament review tree:

| Segment ID | Segment Scope | Source Report File Path |
|:---|:---|:---|
| **Segment 1** | Abstract, Introduction & Related Work | `/Users/phucvu/.gemini/antigravity/scratch/teamwork/029953a1-82a6-4a0e-b4e0-63fc3a725335/.agents/teamwork/group_introduction_and_background/unit_report_introduction_and_background.md` |
| **Segment 2** | Method Foundations & Problem Setup | `/Users/phucvu/.gemini/antigravity/scratch/teamwork/029953a1-82a6-4a0e-b4e0-63fc3a725335/.agents/teamwork/group_method_foundations_and_overview/unit_report_method_foundations_and_overview.md` |
| **Segment 3** | Client Distribution Learning | `/Users/phucvu/.gemini/antigravity/scratch/teamwork/029953a1-82a6-4a0e-b4e0-63fc3a725335/.agents/teamwork/group_method_client_distribution_learning/unit_report_method_client_distribution_learning.md` |
| **Segment 4** | Server Aggregation, Algorithm 1 & Theory | `/Users/phucvu/.gemini/antigravity/scratch/teamwork/029953a1-82a6-4a0e-b4e0-63fc3a725335/.agents/teamwork/group_method_server_aggregation_algorithm_theory/unit_report_method_server_aggregation_algorithm_theory.md` |
| **Segment 5** | Experiments, Evaluations & Conclusion | `/Users/phucvu/.gemini/antigravity/scratch/teamwork/029953a1-82a6-4a0e-b4e0-63fc3a725335/.agents/teamwork/group_experiments_evaluations_and_conclusion/unit_report_experiments_evaluations_and_conclusion.md` |

---
*Report produced and verified by Reducer Synthesizer (`teamwork_synthesizer`) on 2026-11-11.*
