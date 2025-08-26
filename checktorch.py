import torch
print(torch.__version__)          # 파이토치 버전
print(torch.version.cuda)         # CUDA 버전 (없으면 CPU 전용 설치)
print(torch.cuda.is_available())  # True면 GPU 사용 가능

