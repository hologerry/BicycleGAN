# Hyper-paramenters to tune

## model related

1. norm: instance | batch
2. unsample: basic | bilinear
3. nl: relu | lrelu | elu
4. use_attention: true | false
5. init_type: normal | xavier | kaiming | orthogonal

## optimizer related

6. adam beta1: 0.5 | 0.6 | 0.7 | 0.8 | 0.9
7. lr: 0.01 | 0.02 | 0.001 | 0.002 | 0.0001 | 0.0002
8. lr_policy: lambda | step | plateau
9. lr_decay_iters: 100 | 500 | 1000

## lambda parameters

10. lambda_L1: [1.0, 100.0]
11. lambda_L1_B: [1.0, 100.0]
12. lambda_L2: [1.0, 100.0]
13. lambda_CX: [1.0, 100.0]
14. lambda_CX_B: [1.0, 100.0]
15. lambda_GAN: [1.0, 50.0]
16. lambda_GAN_B: [1.0, 50.0]
