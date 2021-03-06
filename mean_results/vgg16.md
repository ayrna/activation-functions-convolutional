# Accuracy (CCR)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | ImageNet | Mean    |
|------------|---------|---------|---------|---------|---------|----------|---------|
| ELU        | 0.81020 | 0.31500 | 0.68468 | 0.99330 | 0.81710 | 0.47104  | 0.68189 |
| ELUs+2     | 0.80690 | 0.33400 | 0.68536 | 0.99342 | 0.82800 | 0.48444  | 0.68869 |
| ELUs+2L    | 0.80770 | 0.31600 | 0.68942 | 0.99318 | 0.82970 | 0.46368  | 0.68328 |
| s+         | 0.79490 | 0.30500 | 0.68951 | 0.99349 | 0.80930 | 0.42746  | 0.66994 |
| s++        | 0.78360 | 0.25000 | 0.65611 | 0.99320 | 0.83100 | 0.43006  | 0.65733 |
| s+2        | 0.80040 | 0.30900 | 0.68572 | 0.99358 | 0.82810 | 0.50848  | 0.68755 |
| s+2L       | 0.80330 | 0.30900 | 0.68032 | 0.99308 | 0.81910 | 0.47576  | 0.68009 |
| EPReLU     | 0.77570 | 0.25900 | 0.60461 | 0.99409 | 0.83330 | 0.44820  | 0.65248 |
| EReLU      | 0.80520 | 0.30100 | 0.65042 | 0.99404 | 0.82940 | 0.05228  | 0.60539 |
| LReLU      | 0.79640 | 0.30100 | 0.65752 | 0.99358 | 0.82810 | 0.47234  | 0.67482 |
| MPELU      | 0.80130 | 0.29300 | 0.66812 | 0.99380 | 0.83460 | 0.46390  | 0.67579 |
| Paired     | 0.78710 | 0.21500 | 0.66591 | 0.99232 | 0.79760 | 0.46342  | 0.65356 |
| PELU       | 0.80400 | 0.08400 | 0.53748 | 0.99276 | 0.73590 | 0.38338  | 0.58959 |
| PReLU      | 0.80960 | 0.27700 | 0.66844 | 0.99309 | 0.82100 | 0.44638  | 0.66925 |
| PTELU      | 0.79910 | 0.27000 | 0.66830 | 0.99271 | 0.82390 | 0.45158  | 0.66760 |
| ReLU       | 0.80210 | 0.26200 | 0.65701 | 0.99396 | 0.82570 | 0.47832  | 0.66985 |
| RReLU      | 0.73620 | 0.17700 | 0.64906 | 0.99162 | 0.81310 | 0.45938  | 0.63773 |
| RTPReLU    | 0.80400 | 0.29300 | 0.66386 | 0.99357 | 0.81910 | 0.49522  | 0.67813 |
| RTReLU     | 0.80800 | 0.28700 | 0.66594 | 0.99296 | 0.82620 | 0.47708  | 0.67620 |
| SlopedReLU | 0.80740 | 0.28400 | 0.65298 | 0.99335 | 0.82810 | 0.47626  | 0.67368 |
| SQRT       | 0.59010 | 0.03300 | 0.36760 | 0.97903 | 0.75360 | 0.28632  | 0.50161 |



# Minimum Sensitivity (MS)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | ImageNet | Mean    |
|------------|---------|---------|---------|---------|---------|----------|---------|
| ELU        | 0.81020 | 0.31500 | 0.68468 | 0.99330 | 0.81710 | 0.06400  | 0.61405 |
| ELUs+2     | 0.80690 | 0.33400 | 0.68536 | 0.99342 | 0.82800 | 0.05200  | 0.61661 |
| ELUs+2L    | 0.80770 | 0.31600 | 0.68942 | 0.99318 | 0.82970 | 0.03200  | 0.61133 |
| s+         | 0.79490 | 0.30500 | 0.68951 | 0.99349 | 0.80930 | 0.03200  | 0.60403 |
| s++        | 0.78360 | 0.25000 | 0.65611 | 0.99320 | 0.83100 | 0.05200  | 0.59432 |
| s+2        | 0.80040 | 0.30900 | 0.68572 | 0.99358 | 0.82810 | 0.04800  | 0.61080 |
| s+2L       | 0.80330 | 0.30900 | 0.68032 | 0.99308 | 0.81910 | 0.05200  | 0.60947 |
| EPReLU     | 0.77570 | 0.25900 | 0.60461 | 0.99409 | 0.83330 | 0.05600  | 0.58712 |
| EReLU      | 0.80520 | 0.30100 | 0.65042 | 0.99404 | 0.82940 | 0.00000  | 0.59668 |
| LReLU      | 0.79640 | 0.30100 | 0.65752 | 0.99358 | 0.82810 | 0.02000  | 0.59943 |
| MPELU      | 0.80130 | 0.29300 | 0.66812 | 0.99380 | 0.83460 | 0.04000  | 0.60514 |
| Paired     | 0.78710 | 0.21500 | 0.66591 | 0.99232 | 0.79760 | 0.06400  | 0.58699 |
| PELU       | 0.80400 | 0.08400 | 0.53748 | 0.99276 | 0.73590 | 0.01200  | 0.52769 |
| PReLU      | 0.80960 | 0.27700 | 0.66844 | 0.99309 | 0.82100 | 0.05600  | 0.60419 |
| PTELU      | 0.79910 | 0.27000 | 0.66830 | 0.99271 | 0.82390 | 0.02000  | 0.59567 |
| ReLU       | 0.80210 | 0.26200 | 0.65701 | 0.99396 | 0.82570 | 0.07600  | 0.60280 |
| RReLU      | 0.73620 | 0.17700 | 0.64906 | 0.99162 | 0.81310 | 0.05200  | 0.56983 |
| RTPReLU    | 0.80400 | 0.29300 | 0.66386 | 0.99357 | 0.81910 | 0.05600  | 0.60492 |
| RTReLU     | 0.80800 | 0.28700 | 0.66594 | 0.99296 | 0.82620 | 0.05600  | 0.60602 |
| SlopedReLU | 0.80740 | 0.28400 | 0.65298 | 0.99335 | 0.82810 | 0.04400  | 0.60164 |
| SQRT       | 0.59010 | 0.03300 | 0.36760 | 0.97903 | 0.75360 | 0.00000  | 0.45389 |


# Mean Absolute Error (MAE)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | ImageNet | Mean    |
|------------|---------|---------|---------|---------|---------|----------|---------|
| ELU        | 0.01668 | 0.00729 | 0.03770 | 0.00071 | 0.01423 | 0.00592  | 0.01375 |
| ELUs+2     | 0.01694 | 0.00739 | 0.03779 | 0.00070 | 0.01447 | 0.00567  | 0.01383 |
| ELUs+2L    | 0.01683 | 0.00724 | 0.03805 | 0.00069 | 0.01399 | 0.00608  | 0.01381 |
| s+         | 0.01776 | 0.00764 | 0.03916 | 0.00073 | 0.01480 | 0.00645  | 0.01442 |
| s++        | 0.01968 | 0.00811 | 0.04095 | 0.00075 | 0.01429 | 0.00639  | 0.01503 |
| s+2        | 0.01777 | 0.00757 | 0.03866 | 0.00073 | 0.01459 | 0.00552  | 0.01414 |
| s+2L       | 0.01766 | 0.00740 | 0.03882 | 0.00075 | 0.01480 | 0.00578  | 0.01420 |
| EPReLU     | 0.02181 | 0.00852 | 0.04547 | 0.00088 | 0.01574 | 0.00630  | 0.01645 |
| EReLU      | 0.01919 | 0.00796 | 0.04187 | 0.00077 | 0.01429 | 0.00979  | 0.01564 |
| LReLU      | 0.01798 | 0.00738 | 0.04008 | 0.00070 | 0.01475 | 0.00594  | 0.01447 |
| MPELU      | 0.01721 | 0.00737 | 0.03828 | 0.00068 | 0.01375 | 0.00601  | 0.01388 |
| Paired     | 0.01884 | 0.00865 | 0.04059 | 0.00076 | 0.01557 | 0.00606  | 0.01508 |
| PELU       | 0.01770 | 0.01553 | 0.06810 | 0.00072 | 0.03072 | 0.00684  | 0.02327 |
| PReLU      | 0.01743 | 0.00743 | 0.03912 | 0.00071 | 0.01380 | 0.00626  | 0.01413 |
| PTELU      | 0.01714 | 0.00788 | 0.03921 | 0.00070 | 0.01348 | 0.00625  | 0.01411 |
| ReLU       | 0.01752 | 0.00769 | 0.03909 | 0.00067 | 0.01385 | 0.00572  | 0.01409 |
| RReLU      | 0.02228 | 0.00902 | 0.04174 | 0.00073 | 0.01532 | 0.00611  | 0.01587 |
| RTPReLU    | 0.01774 | 0.00742 | 0.03918 | 0.00067 | 0.01478 | 0.00564  | 0.01424 |
| RTReLU     | 0.01761 | 0.00747 | 0.03996 | 0.00068 | 0.01399 | 0.00585  | 0.01426 |
| SlopedReLU | 0.01727 | 0.00749 | 0.03998 | 0.00071 | 0.01425 | 0.00595  | 0.01427 |
| SQRT       | 0.04729 | 0.01348 | 0.07365 | 0.00200 | 0.02554 | 0.00809  | 0.02834 |



# Mean Squared Error (MSE)
| Activation | CIF-10  | CIF-100 | CIN-10  | MNIST   | Fashion | ImageNet | Mean    |
|------------|---------|---------|---------|---------|---------|----------|---------|
| ELU        | 0.01290 | 0.00490 | 0.02409 | 0.00049 | 0.00918 | 0.00350  | 0.00918 |
| ELUs+2     | 0.01303 | 0.00488 | 0.02394 | 0.00049 | 0.00913 | 0.00346  | 0.00916 |
| ELUs+2L    | 0.01295 | 0.00493 | 0.02400 | 0.00047 | 0.00905 | 0.00350  | 0.00915 |
| s+         | 0.01330 | 0.00489 | 0.02433 | 0.00051 | 0.00933 | 0.00366  | 0.00934 |
| s++        | 0.01383 | 0.00499 | 0.02501 | 0.00054 | 0.00921 | 0.00369  | 0.00955 |
| s+2        | 0.01330 | 0.00495 | 0.02421 | 0.00050 | 0.00918 | 0.00326  | 0.00923 |
| s+2L       | 0.01329 | 0.00497 | 0.02418 | 0.00052 | 0.00929 | 0.00353  | 0.00930 |
| EPReLU     | 0.01429 | 0.00505 | 0.02571 | 0.00053 | 0.00932 | 0.00353  | 0.00974 |
| EReLU      | 0.01377 | 0.00508 | 0.02432 | 0.00048 | 0.00887 | 0.00490  | 0.00957 |
| LReLU      | 0.01297 | 0.00481 | 0.02423 | 0.00048 | 0.00927 | 0.00347  | 0.00920 |
| MPELU      | 0.01366 | 0.00512 | 0.02448 | 0.00051 | 0.00914 | 0.00350  | 0.00940 |
| Paired     | 0.01481 | 0.00531 | 0.02545 | 0.00058 | 0.01011 | 0.00349  | 0.00996 |
| PELU       | 0.01365 | 0.00827 | 0.03760 | 0.00054 | 0.01735 | 0.00395  | 0.01356 |
| PReLU      | 0.01360 | 0.00507 | 0.02445 | 0.00053 | 0.00890 | 0.00358  | 0.00935 |
| PTELU      | 0.01339 | 0.00509 | 0.02420 | 0.00052 | 0.00904 | 0.00354  | 0.00930 |
| ReLU       | 0.01373 | 0.00506 | 0.02451 | 0.00050 | 0.00915 | 0.00352  | 0.00941 |
| RReLU      | 0.01710 | 0.00585 | 0.02663 | 0.00054 | 0.01016 | 0.00352  | 0.01063 |
| RTPReLU    | 0.01374 | 0.00503 | 0.02437 | 0.00049 | 0.00939 | 0.00333  | 0.00939 |
| RTReLU     | 0.01380 | 0.00500 | 0.02448 | 0.00050 | 0.00904 | 0.00344  | 0.00938 |
| SlopedReLU | 0.01357 | 0.00509 | 0.02477 | 0.00054 | 0.00923 | 0.00340  | 0.00943 |
| SQRT       | 0.02758 | 0.00711 | 0.03941 | 0.00121 | 0.01356 | 0.00427  | 0.01552 |