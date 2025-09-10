from IPython import get_ipython
get_ipython().magic('clear')

import numpy as np
import scipy.io
import os

project_name = "phy_joint_jpic";
path_folder = os.path.abspath(os.path.dirname(__file__)).lower();
path_folder = path_folder[:path_folder.find(project_name)+len(project_name)];
path_file = os.path.normpath(path_folder+"/_tmp/tests/theory/lmmse_mask_iter_1.mat")



# load matlab data
matlab_data = scipy.io.loadmat(path_file);
nt          = np.squeeze(matlab_data["nt"]);
nr          = np.squeeze(matlab_data["nr"]);
mask        = matlab_data["mask"];
mask_row    = matlab_data["mask_row"];
x           = matlab_data["x"];
x_hat1      = matlab_data["x_hat1"];
x_hat1_m    = matlab_data["x_hat1_m"];
x_hat2      = matlab_data["x_hat2"];
x_hat2_m    = matlab_data["x_hat2_m"];
H           = matlab_data["H"];
z           = matlab_data["z"];
y           = matlab_data["y"];
No          = np.squeeze(matlab_data["No"]);

'''
LMMSE mask H
'''
Hm = H * mask_row;
H_pinv = np.linalg.solve(Hm.T.conj() @ Hm + No*np.eye(nt), Hm.T.conj());
py_x_hat1 = H_pinv @ y;
py_x_hat1_m = mask * H_pinv @ y;
# compare
diff_x_hat1 = abs(x_hat1 - py_x_hat1)
diff_x_hat1_m = abs(x_hat1_m - py_x_hat1_m)
print("- LMMSE mask H");
print("  - x_hat diff %e"%np.max(diff_x_hat1))
print("  - x_hat diff (mask output) %e"%np.max(diff_x_hat1_m))

'''
LMMSE mask x
'''
H_pinv = np.linalg.solve((H.T.conj()@H)*mask_row + No*np.eye(nt), H.T.conj());
py_x_hat2 = H_pinv @ y;
py_x_hat2_m = mask * H_pinv @ y;
# compare
diff_x_hat2 = abs(x_hat2 - py_x_hat2)
diff_x_hat2_m = abs(x_hat2_m - py_x_hat2_m)
print("- LMMSE mask H");
print("  - x_hat diff %e"%np.max(diff_x_hat2))
print("  - x_hat diff (mask output) %e"%np.max(diff_x_hat2_m))
