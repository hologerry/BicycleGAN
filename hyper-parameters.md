# Hyper-paramenters to tune

## model related

1. norm: instance | batch (0, 1)
2. upsample: basic | bilinear (0, 1)
3. nl: relu | lrelu | elu (0, 1, 2)
4. use_attention: false | true (0, 1)
5. init_type: normal | xavier | kaiming | orthogonal (0, 1, 2, 3)

## optimizer related

6. adam beta1: 0.5 | 0.6 | 0.7 | 0.8 | 0.9 (0, 1, 2, 3, 4)
7. lr: 0.01 | 0.02 | 0.001 | 0.002 | 0.0001 | 0.0002 (0, 1, 2, 3, 4, 5)
8. lr_policy: lambda | step | plateau (0, 1, 2)
9. lr_decay_iters: 100 | 500 | 1000 (0, 1, 2)

## lambda parameters

10. lambda_L1: [1.0, 100.0] (0, ..., 19) * 5
11. lambda_L1_B: [1.0, 100.0] (0, ..., 19)
12. lambda_L2: [1.0, 100.0] (0, ..., 19)
13. lambda_CX: [1.0, 100.0] (0, ..., 19)
14. lambda_CX_B: [1.0, 100.0] (0, ..., 19)
15. lambda_GAN: [1.0, 50.0] (0, ..., 9)
16. lambda_GAN_B: [1.0, 50.0] (0, ..., 9)
