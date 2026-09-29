# Invalidated float32 inverse-CDF run

`compact_cnn_ags_clip_screen_b2000.json` is retained for auditability but must
not be interpreted as an AGS utility result. The float32 uniform variates were
clamped to `1e-12`, which is too small to remain distinct from zero after the
subtraction used by the inverse Laplace CDF. An exact zero RNG output could
therefore produce infinite noise. Every cap reused the same noise seed and was
poisoned by the same endpoint draw.

The implementation now clamps to `torch.finfo(dtype).eps`; the corrected rerun
uses the `_v2` artifact suffix.

