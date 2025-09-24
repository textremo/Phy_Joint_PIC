from IPython import get_ipython
get_ipython().magic('clear')

import numpy as np
import scipy.io
import os

project_name = "phy_joint_jpic";
path_folder = os.path.abspath(os.path.dirname(__file__)).lower();
path_folder = path_folder[:path_folder.find(project_name)+len(project_name)];
path_file = os.path.normpath(path_folder+"/_tmp/tests/theory/lmmse_mask_iter_n.mat")


# load matlab data
matlab_data = scipy.io.loadmat(path_file)
trials      = np.squeeze(matlab_data["trials"])
nt          = np.squeeze(matlab_data["nt"])
nr          = np.squeeze(matlab_data["nr"])
masks       = matlab_data["masks"][..., None]
mask_rows   = matlab_data["mask_rows"]
xs          = matlab_data["xs"][..., None]
x_hat1s     = matlab_data["x_hat1s"][..., None]
x_hat1_ms   = matlab_data["x_hat1_ms"][..., None]
x_hat2s     = matlab_data["x_hat2s"][..., None]
x_hat2_ms   = matlab_data["x_hat2_ms"][..., None]
Hs          = matlab_data["Hs"]
ys          = matlab_data["ys"][..., None]
No          = np.squeeze(matlab_data["No"])
mse1        = np.squeeze(matlab_data["mse1"])
mse1_m      = np.squeeze(matlab_data["mse1_m"])
mse2        = np.squeeze(matlab_data["mse2"])
mse2_m      = np.squeeze(matlab_data["mse2_m"])


err1 = np.zeros([nt, 1]);
err1_m = np.zeros([nt, 1]);
err2 = np.zeros([nt, 1]);
err2_m = np.zeros([nt, 1]);

for i in range(trials):
    mask = masks[i, ...]
    mask_row = mask_rows[i, ...]
    x = xs[i, ...]
    H = Hs[i, ...]
    y = ys[i, ...]
    
    mat_x_hat1 = x_hat1s[i, ...]
    mat_x_hat1_m = x_hat1_ms[i, ...]
    mat_x_hat2 = x_hat2s[i, ...]
    mat_x_hat2_m = x_hat2_ms[i, ...]
    
    '''
    LMMSE mask H
    '''
    Hm = H * mask_row;
    H_pinv = np.linalg.solve(Hm.T.conj() @ Hm + No*np.eye(nt), Hm.T.conj());
    py_x_hat1 = H_pinv @ y;
    py_x_hat1_m = mask * H_pinv @ y;
    err1 = err1 + abs(py_x_hat1 - x)**2; 
    err1_m = err1_m + abs(py_x_hat1_m - x)**2; 
    # compare
    diff_x_hat1 = abs(mat_x_hat1 - py_x_hat1)
    diff_x_hat1_m = abs(mat_x_hat1_m - py_x_hat1_m)
    if np.max(diff_x_hat1) > 1e-10 or np.max(diff_x_hat1_m) > 1e-10:
        print("- LMMSE mask H iter: %d"%i);
        print("  - x_hat diff %e"%np.max(diff_x_hat1))
        print("  - x_hat diff (mask output) %e"%np.max(diff_x_hat1_m))
    
    
    '''
    LMMSE mask x
    '''
    H_pinv = np.linalg.solve((H.T.conj()@H)*mask_row + No*np.eye(nt), H.T.conj());
    py_x_hat2 = H_pinv @ y;
    py_x_hat2_m = mask * H_pinv @ y;
    err2 = err2 + abs(py_x_hat2 - x)**2; 
    err2_m = err2_m + abs(py_x_hat2_m - x)**2; 
    # compare
    diff_x_hat2 = abs(mat_x_hat2 - py_x_hat2)
    diff_x_hat2_m = abs(mat_x_hat2_m - py_x_hat2_m)
    if np.max(diff_x_hat2) > 1e-10 or np.max(diff_x_hat2_m) > 1e-10:
        print("- LMMSE mask H iter: %d"%i);
        print("  - x_hat diff %e"%np.max(diff_x_hat2))
        print("  - x_hat diff (mask output) %e"%np.max(diff_x_hat2_m))
        
py_mse1 = np.mean(err1/trials);
py_mse1_m = np.mean(err1_m/trials);
print("- LMMSE mask H");
print("  - x_hat MSE diff: %e"%(py_mse1 - mse1));
print("  - x_hat MSE mask diff): %e"%(py_mse1_m - mse1_m));

py_mse2 = np.mean(err2/trials);
py_mse2_m = np.mean(err2_m/trials);
print("- LMMSE mask x");
print("  - x_hat MSE diff: %e"%(py_mse2 - mse2));
print("  - x_hat MSE mask diff): %e"%(py_mse2_m - mse2_m));