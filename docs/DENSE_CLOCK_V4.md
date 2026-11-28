# One additional trained clock

Run the 14-bin GRU at a one-bin stride, giving a trained 20 ms emission interval. This completes a 20, 40 and 80 ms stride comparison on the recorded 20 ms features. The earlier dense-stride result changed only the frozen checkpoint.

Use the same seed, two trainable recurrent layers, head, official training order, 2,048-update endpoint, 900-second pilot and beam-50 word decoder as the primary GRU arms. The finer stride increases the number of available CTC frames and leaves the common training trial feasibility filter unchanged. Check the saved membership hash against the primary arms. This arm was added after the frozen-clock diagnostic and before examining any trained-arm endpoint.
