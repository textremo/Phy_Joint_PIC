'''
check the status of all packages
'''
import torch
if torch.cuda.is_available():
    print("PyTorch is using GPU")
else:
    print("PyTorch is not using GPU")
    
device_num = torch.cuda.device_count();
for i in range(device_num):
    name = torch.cuda.get_device_name(i);
    free_mem, total_mem = torch.cuda.mem_get_info(i)
      
    print("No.%d: %s"%(i, name))
    print(f"  - Memory (available): {free_mem / 1024**3:.2f} GB")
    print(f"  - Memory (total):     {total_mem / 1024**3:.2f} GB")
    print(f"  - Memory (used):      {(total_mem - free_mem) / 1024**3:.2f} GB")
    
    
    # print(" - torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(i)/1024/1024/1024))
    # print(" - torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(i)/1024/1024/1024))
    # print(" - torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(i)/1024/1024/1024))