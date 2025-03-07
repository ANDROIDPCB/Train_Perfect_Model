import torch


def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device