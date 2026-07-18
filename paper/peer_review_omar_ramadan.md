# First Draft's Peer Review

## Measuring the Tightness Gap: An Empirical Audit of Differential Privacy Guarantees in DP-SGD

**Student:** Mohamad Faour (msf20)
**Reviewer:** Omar Ramadan (ID: 202204622)
**Course:** EECE 608 - Trustworthy Machine Learning
**Term:** Spring 2025 - 2026
**Review Submission Date:** Tuesday, April 7th, 2026

---

## Summary

This paper investigates the tightness gap between the theoretical privacy guarantees produced by DP-SGD's privacy accountant and the leakage an adversary can actually demonstrate empirically. The author builds an end-to-end auditing pipeline evaluated under two threat models: an active canary insertion attack (where crafted examples are planted in the training set before training) and a passive membership inference attack (where the adversary is limited to querying the released model). A small two-layer MLP is trained on MNIST using the Opacus library with a Renyi DP accountant yielding a theoretical upper bound of epsilon = 0.771. The best canary auditor recovers roughly 36% of that bound, while the strongest non-pathological passive result reaches about 12%. The paper is sincere about implementation issues encountered along the way (seed propagation bug, pathological small-sample bounds, score direction misalignment in passive pipeline), and it documents the fixes applied to each. The work is framed explicitly as an engineering validation phase, with a clear list of planned next steps, aiming toward stronger attacks and more statistically grounded conclusions for the final submission.

## Strengths

The strongest part of this first draft is arguably Section 5, which documents the bugs encountered during development. In a lot of student projects (and even some real and polished publications), things that break or fail are rarely even mentioned, but here, the author goes out of their way to explain what went wrong (seed propagation errors, pathological bounds from sparse tails, score direction misalignment) and what was done to fix each one, and this kind of transparency actually strengthens the credibility of the results rather than weakening them. Another strong item is the dual-track design (active canary vs. passive membership inference): keeping the two pipelines separate in code and configuration is good practice and makes the comparison clean and fair. Moreover, the use of Wilson score intervals for conservative bound estimation shows that the author is thinking carefully about statistical reliability (which is not something everyone building these pipelines gets right on the first try). The writing is generally clear and the tables are well-formatted and easy to read, and the next steps section is concrete, actionable and doable rather than vague and unrealistic.

## Weaknesses

The most notable weakness is the instability of the passive auditing results: even after applying the postfix score direction correction, several seed-variant combinations still produce pathological bounds exceeding the theoretical upper limit (e.g., probability margin on seed 125 yields a ratio of 179.8%, and both max probability and negative loss on seed 123 reach 142.5%), which makes it hard to draw reliable conclusions from that track. The author mentions the small query budget (128 examples) as a cause, but the persistence of these artifacts after the fixes suggests that the passive pipeline may need more fundamental reassessment instead. A second weakness is the complete absence of figures or visualizations: the paper relies entirely on tables, and something like score distribution plots for members vs. non-members (or an ROC curve with the theoretical epsilon bound) would make the empirical story considerably more intuitive and conforming to auditing papers. Finally, the Related Work section is placed near the end (Section 6) rather than early on where it would give the reader the necessary context before proceeding with the Methodology section.

## Recommendations for Final Submission

- Use your AUB institutional email address on the paper instead of a personal Gmail (small detail but matters for academic submissions).
- Move Section 6 (Related Work) to appear right after the introduction, so that the reader has the necessary information before diving into the methodology.
- Add at least one figure (e.g., score distribution plots for members vs. non-members, ROC-style curve with the theoretical epsilon bound) to make the empirical story much more convincing.
- Table 4 appears in the paper but is never explicitly referenced or discussed in the text. Even a single sentence pointing the reader to it and briefly interpreting the cross-track comparison would close that gap.
- The three training seeds (123, 124, 125) are consecutive integers. Some pseudorandom generators have reduced independence when seeds are numerically close to each other. It is probably fine at this scale, but using more spread-out seeds (e.g., 42, 123, and 456) would be a safer and more standard choice. Either way, make sure of the pseudorandom algorithm used by your coding language or imported library.
- For the final submission, adopt a cleaner NeurIPS paper writing format. The current progress-report style works for a first draft but will need to be restructured.

Overall this is a well-executed first draft with honest reporting and a clear direction forward. The engineering work is solid and the transparency around what broke and how it was fixed is genuinely commendable. Addressing the instability in the passive track and adding visualizations would significantly strengthen the final submission.
