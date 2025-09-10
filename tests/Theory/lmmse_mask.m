clear;
clc;

nr = 100;
nt = 40;
SNR = 20;
No = 10^(-SNR/10);
M_mod = 4;
constel = qammod(0: M_mod-1, M_mod, 'UnitAveragePower',true).';
trials = 1e4;

err1 = zeros(nt, 1);
err1_m = zeros(nt, 1);
err2 = zeros(nt, 1);
err2_m = zeros(nt, 1);
for iter = 1:trials
    
    x_idx = randi(M_mod, nt, 1);
    x = constel(x_idx);
    %x = sqrt(1/2)*(randn(nt, 1) + 1j*randn(nt, 1));
    mask = rand(nt, 1) > 0.4;
    mask_row = mask.';
    x = mask.*x;

    H = sqrt(1/2/nt)*(randn(nr, nt) + 1j*randn(nr, nt));
    z = sqrt(No/2)*(randn(nr, 1) + 1j*randn(nr, 1));
    y = H*x + z;

    %% LMMSE mask H
    Hm = H.*mask_row;
    x_hat1 = (Hm'*Hm + No.*eye(nt)) \ Hm'*y;
    x_hat1_m = mask.*((Hm'*Hm + No*eye(nt)) \ Hm'*y);
    err1 = err1 + abs(x_hat1 - x).^2; 
    err1_m = err1_m + abs(x_hat1_m - x).^2; 
    
    %% LMMSE mask x
    x_hat2 = (H'*H.*mask_row + No.*eye(nt)) \ H'*y;
    x_hat2_m = mask.*((H'*H.*mask_row + No*eye(nt)) \ (H'*y));
    err2 = err2 + abs(x_hat2 - x).^2; 
    err2_m = err2_m + abs(x_hat2_m - x).^2; 

end

mse1 = mean(err1/trials);
mse1_m = mean(err1_m/trials);
mse2 = mean(err2/trials);
mse2_m = mean(err2_m/trials);
