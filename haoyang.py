import time

import torch

def main():
    device = torch.device("cuda:0")
    x = torch.zeros((256, 256), device=device)

    print("已在 GPU 上分配一小块显存。按 Ctrl+C 退出程序。")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("退出，显存会被自动释放。")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 CUDA GPU。")
    main()
