Traditional baseline polarity fix v2

Reason for revision:
The training-sanity run showed that the raw-intensity THRESHOLD baselines
(B0 and B3) were selecting the bright pavement/background as foreground.
This is a semantic foreground-polarity error, not performance tuning.

Only change:
- B0: add BITWISE_NOT immediately after THRESHOLD.
- B3: add BITWISE_NOT immediately after THRESHOLD.
- B1 and B2: unchanged.
- All numeric parameters: unchanged.

Rerun TRAINING SANITY (111-130) with these trees.
Do NOT run primary200 yet.

If sanity is then structurally plausible (binary outputs, no obvious inversion,
no empty-all-images behavior caused by polarity), freeze this v2 set before
primary200.

SHA-256:
e03ccbb133931efefa0a5cb53b9140c0fc3b8f549e3a9483607934b31c7bd979  baseline_B0_gaussian_threshold_printed_tree_sys.txt
70737781ab88e0777c86526565d72442d0b798c1f367953f36baa491ad92584d  baseline_B1_canny_morph_cc_printed_tree_sys.txt
cd039253205a7ceec8050a5b58c6b65da578506abc53711d3de21461ab9290c1  baseline_B2_sobel_xy_threshold_morph_cc_printed_tree_sys.txt
d491964d592a3602a8f956dd6116009518243a119137dcd982ffc3c074066bf4  baseline_B3_median_threshold_opening_cc_printed_tree_sys.txt
