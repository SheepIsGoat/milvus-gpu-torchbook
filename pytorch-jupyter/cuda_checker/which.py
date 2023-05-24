import torch
import socket
from typing import List

def dprint(
        *args
) -> None:
    skip = all([
        arg for arg in args
        if isinstance(arg, bool)
    ])

    if skip:
        return
    
    res = "".join([
        arg for arg in args if not isinstance(arg, bool)
    ])
    print(res)

def which_cuda(
    verbose: bool=True
) -> List[str]:
    """
    Show which cuda devices are recognized by pytorch on your device.
    """
    dprint(
        "Checking torch access to CUDA GPUs with hostname", 
        socket.gethostname(),
        verbose
    )
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        dprint(
            f"GOOD: CUDA is available with {num_devices} device{'s' if num_devices!=1 else ''}",
            verbose
        )
        devices = [
            torch.cuda.get_device_name(idx)
            for idx in range(num_devices)
        ]
        dprint(
            "DEVICES:", 
            devices,
            verbose
        )
        return devices
    else:
        dprint(
            "NO DEVICES FOUND.",
            verbose
        )
        return []

if __name__=="__main__":
    which_cuda()