import os
import torch
import struct
import subprocess
import argparse
from pathlib import Path

def git_describe():
    # return human-readable git description, i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe
    if Path('.git').exists():
        return subprocess.check_output('git describe --tags --long --always', shell=True).decode('utf-8')[:-1]
    else:
        return ''


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 {git_describe()} torch {torch.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    return torch.device('cuda:0' if cuda else 'cpu')
    


def Run():
    # Initialize
    device = select_device('cpu')
    # Load model
    model = torch.load(opt.weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    
    f = open(opt.savewts, 'w')
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

    
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov5l.pt', help='model.pt path')
    parser.add_argument('--savewts', type=str, default='yolov5l.wts')
    opt = parser.parse_args()
    print(opt)
    
    Run()
    print('Done!')