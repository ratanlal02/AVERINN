\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xbb6960a0adc995a4
Maximize
  x_1
Subject To
 R0: x_0 - alpha_0 - 2.125 alpha_1 >= 5.375
 R1: x_1 - alpha_0 + 2.125 alpha_1 - 4.25 alpha_2 >= 5.625
 R2: x_0 - alpha_0 - 2.125 alpha_1 <= 5.375
 R3: x_1 - alpha_0 + 2.125 alpha_1 - 4.25 alpha_2 <= 5.625
 R4: alpha_0 <= 1
 R5: - alpha_0 <= 1
 R6: alpha_1 <= 1
 R7: - alpha_1 <= 1
 R8: alpha_2 <= 1
 R9: - alpha_2 <= 1
Bounds
 x_0 free
 x_1 free
 alpha_0 free
 alpha_1 free
 alpha_2 free
End
