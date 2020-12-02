# Accuracy (CCR)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| ELU        | 0.75824 | 0.47647 | 0.67661 | 0.99341 | 0.90886 | 0.76272 |
| ELUs+2     | 0.76271 | 0.48172 | 0.67827 | 0.99395 | 0.90912 | 0.76515 |
| ELUs+2L    | 0.75814 | 0.47920 | 0.67618 | 0.99398 | 0.90581 | 0.76266 |
| s+         | 0.75455 | 0.46892 | 0.66859 | 0.99347 | 0.90774 | 0.75865 |
| s+         | 0.73985 | 0.45705 | 0.65253 | 0.99424 | 0.90852 | 0.75044 |
| s+2        | 0.74799 | 0.46066 | 0.66375 | 0.99346 | 0.90517 | 0.75421 |
| s+2L       | 0.75038 | 0.46215 | 0.66460 | 0.99324 | 0.90581 | 0.75524 |
| EPReLU     | 0.74595 | 0.47036 | 0.66558 | 0.95210 | 0.83222 | 0.73324 |
| EReLU      | 0.67731 | 0.33190 | 0.63251 | 0.95618 | 0.68884 | 0.65735 |
| LReLU      | 0.74565 | 0.45028 | 0.66468 | 0.99356 | 0.90561 | 0.75196 |
| MPELU      | 0.68177 | 0.35498 | 0.63690 | 0.99293 | 0.89867 | 0.71305 |
| Paired     | 0.75286 | 0.47714 | 0.67555 | 0.99326 | 0.90622 | 0.76101 |
| PELU       | 0.71838 | 0.35550 | 0.59991 | 0.99291 | 0.90379 | 0.71410 |
| PReLU      | 0.67712 | 0.35432 | 0.63447 | 0.99319 | 0.89966 | 0.71175 |
| PTELU      | 0.72445 | 0.41755 | 0.66160 | 0.99367 | 0.90408 | 0.74027 |
| ReLU       | 0.67865 | 0.35419 | 0.63537 | 0.99349 | 0.89929 | 0.71220 |
| RReLU      | 0.49276 | 0.14595 | 0.53366 | 0.98241 | 0.74584 | 0.58012 |
| RTPReLU    | 0.73425 | 0.43293 | 0.66086 | 0.99341 | 0.90486 | 0.74526 |
| RTReLU     | 0.71158 | 0.40131 | 0.64863 | 0.99356 | 0.90335 | 0.73169 |
| SlopedReLU | 0.44290 | 0.17503 | 0.35846 | 0.61885 | 0.74268 | 0.46758 |
| SQRT       | 0.38349 | 0.13135 | 0.35178 | 0.90126 | 0.75537 | 0.50465 |


# Minimum Sensitivity (MS)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| ELU        | 0.52310 | 0.10800 | 0.43437 | 0.98708 | 0.73520 | 0.55755 |
| ELUs+2     | 0.53540 | 0.10500 | 0.45343 | 0.98735 | 0.75050 | 0.56634 |
| ELUs+2L    | 0.50840 | 0.10500 | 0.44480 | 0.98825 | 0.73100 | 0.55549 |
| s+         | 0.53150 | 0.09200 | 0.43670 | 0.98563 | 0.72020 | 0.55321 |
| s+         | 0.49010 | 0.07500 | 0.41919 | 0.98935 | 0.74310 | 0.54335 |
| s+2        | 0.52250 | 0.10000 | 0.43174 | 0.98729 | 0.73450 | 0.55521 |
| s+2L       | 0.51100 | 0.08900 | 0.42246 | 0.98577 | 0.72580 | 0.54681 |
| EPReLU     | 0.43490 | 0.07600 | 0.35969 | 0.79233 | 0.63350 | 0.45928 |
| EReLU      | 0.31620 | 0.01900 | 0.32596 | 0.79656 | 0.30220 | 0.35198 |
| LReLU      | 0.49300 | 0.09000 | 0.42269 | 0.98730 | 0.72420 | 0.54344 |
| MPELU      | 0.44890 | 0.05000 | 0.39062 | 0.98585 | 0.71260 | 0.51759 |
| Paired     | 0.50110 | 0.07000 | 0.42867 | 0.98688 | 0.72000 | 0.54133 |
| PELU       | 0.48080 | 0.05000 | 0.39041 | 0.98532 | 0.70650 | 0.52261 |
| PReLU      | 0.44070 | 0.04300 | 0.40018 | 0.98507 | 0.70620 | 0.51503 |
| PTELU      | 0.46910 | 0.08500 | 0.42450 | 0.98617 | 0.71610 | 0.53617 |
| ReLU       | 0.43160 | 0.03400 | 0.39621 | 0.98674 | 0.70980 | 0.51167 |
| RReLU      | 0.23310 | 0.00000 | 0.27502 | 0.96741 | 0.38850 | 0.37281 |
| RTPReLU    | 0.49610 | 0.06800 | 0.42656 | 0.98625 | 0.71250 | 0.53788 |
| RTReLU     | 0.44870 | 0.07300 | 0.40804 | 0.98658 | 0.70970 | 0.52520 |
| SlopedReLU | 0.25390 | 0.01800 | 0.17909 | 0.49118 | 0.45720 | 0.27987 |
| SQRT       | 0.14800 | 0.00000 | 0.17927 | 0.77084 | 0.25860 | 0.27134 |


# Mean Absolute Error (MAE)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| ELU        | 0.05822 | 0.01216 | 0.08082 | 0.00165 | 0.02314 | 0.03520 |
| ELUs+2     | 0.05698 | 0.01226 | 0.08070 | 0.00164 | 0.02330 | 0.03498 |
| ELUs+2L    | 0.05818 | 0.01220 | 0.08075 | 0.00168 | 0.02386 | 0.03533 |
| s+         | 0.06047 | 0.01253 | 0.08401 | 0.00181 | 0.02404 | 0.03657 |
| s+         | 0.06551 | 0.01289 | 0.08810 | 0.00132 | 0.02412 | 0.03839 |
| s+2        | 0.06054 | 0.01284 | 0.08398 | 0.00198 | 0.02435 | 0.03674 |
| s+2L       | 0.06162 | 0.01278 | 0.08338 | 0.00200 | 0.02426 | 0.03681 |
| EPReLU     | 0.06366 | 0.01273 | 0.08720 | 0.01323 | 0.04629 | 0.04462 |
| EReLU      | 0.07787 | 0.01538 | 0.09362 | 0.01090 | 0.07475 | 0.05450 |
| LReLU      | 0.06142 | 0.01282 | 0.08360 | 0.00171 | 0.02408 | 0.03673 |
| MPELU      | 0.07740 | 0.01491 | 0.08957 | 0.00186 | 0.02581 | 0.04191 |
| Paired     | 0.05998 | 0.01259 | 0.08085 | 0.00205 | 0.02402 | 0.03590 |
| PELU       | 0.06914 | 0.01474 | 0.09490 | 0.00176 | 0.02432 | 0.04097 |
| PReLU      | 0.07841 | 0.01495 | 0.09044 | 0.00178 | 0.02569 | 0.04225 |
| PTELU      | 0.06635 | 0.01339 | 0.08417 | 0.00164 | 0.02455 | 0.03802 |
| ReLU       | 0.07764 | 0.01496 | 0.09058 | 0.00171 | 0.02569 | 0.04212 |
| RReLU      | 0.12512 | 0.01871 | 0.11805 | 0.00508 | 0.06486 | 0.06636 |
| RTPReLU    | 0.06443 | 0.01339 | 0.08494 | 0.00174 | 0.02458 | 0.03782 |
| RTReLU     | 0.07034 | 0.01401 | 0.08784 | 0.00171 | 0.02470 | 0.03972 |
| SlopedReLU | 0.12073 | 0.01750 | 0.13850 | 0.07642 | 0.05576 | 0.08178 |
| SQRT       | 0.15078 | 0.01899 | 0.15427 | 0.03110 | 0.06525 | 0.08408 |


# Mean Squared Error (MSE)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | Mean    |
|------------|---------|---------|---------|---------|---------|---------|
| ELU        | 0.03467 | 0.00676 | 0.04359 | 0.00098 | 0.01336 | 0.01987 |
| ELUs+2     | 0.03401 | 0.00661 | 0.04334 | 0.00093 | 0.01342 | 0.01966 |
| ELUs+2L    | 0.03456 | 0.00667 | 0.04361 | 0.00093 | 0.01374 | 0.01990 |
| s+         | 0.03463 | 0.00674 | 0.04438 | 0.00099 | 0.01353 | 0.02005 |
| s+         | 0.03652 | 0.00690 | 0.04636 | 0.00096 | 0.01347 | 0.02084 |
| s+2        | 0.03592 | 0.00681 | 0.04517 | 0.00100 | 0.01390 | 0.02056 |
| s+2L       | 0.03535 | 0.00679 | 0.04513 | 0.00104 | 0.01384 | 0.02043 |
| EPReLU     | 0.03566 | 0.00673 | 0.04471 | 0.00710 | 0.02358 | 0.02356 |
| EReLU      | 0.04488 | 0.00808 | 0.04872 | 0.00657 | 0.04273 | 0.03020 |
| LReLU      | 0.03632 | 0.00696 | 0.04502 | 0.00098 | 0.01388 | 0.02063 |
| MPELU      | 0.04415 | 0.00783 | 0.04843 | 0.00110 | 0.01476 | 0.02325 |
| Paired     | 0.03525 | 0.00662 | 0.04385 | 0.00104 | 0.01381 | 0.02011 |
| PELU       | 0.03947 | 0.00871 | 0.05957 | 0.00109 | 0.01419 | 0.02461 |
| PReLU      | 0.04471 | 0.00784 | 0.04871 | 0.00106 | 0.01465 | 0.02339 |
| PTELU      | 0.03912 | 0.00734 | 0.04540 | 0.00097 | 0.01411 | 0.02139 |
| ReLU       | 0.04462 | 0.00784 | 0.04861 | 0.00102 | 0.01472 | 0.02336 |
| RReLU      | 0.06387 | 0.00936 | 0.05967 | 0.00269 | 0.03489 | 0.03410 |
| RTPReLU    | 0.03765 | 0.00710 | 0.04542 | 0.00101 | 0.01390 | 0.02102 |
| RTReLU     | 0.04043 | 0.00740 | 0.04694 | 0.00100 | 0.01414 | 0.02198 |
| SlopedReLU | 0.09786 | 0.01388 | 0.11343 | 0.07603 | 0.04693 | 0.06963 |
| SQRT       | 0.07482 | 0.00947 | 0.07683 | 0.01474 | 0.03296 | 0.04176 |