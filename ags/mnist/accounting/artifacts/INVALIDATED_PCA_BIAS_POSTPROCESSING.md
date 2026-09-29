# Invalidated PCA public-bias postprocessing artifact

`pca_early_stop_strong_overnight.json` used additive public class intercepts
without a fitted logit temperature. For early PCA iterates the logits are small
but correctly ordered, so the intercepts overwhelmed their margins and reduced
accuracy toward chance. The raw AGS and DP-SGD releases in that artifact remain
valid, but all `public_class_bias_recalibrated_*` fields are invalid diagnostics.

`pca_early_stop_strong_corrected_overnight.json` supersedes it. The corrected
postprocessing fits a positive public logit temperature and class intercepts,
then applies that affine transformation to the already-private released logits.
