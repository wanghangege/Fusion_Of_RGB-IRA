class args():

    # hyperparameters
    put_type = 'mean'
    balance = 0.01

    # training args
    epochs = 300 # "number of training epochs, default is 2"
    save_per_epoch = 1
    batch_size = 1 # "batch size for training/testing, default is 4"
    dataset1 = "./data/train_visible.txt"
    dataset2 = "./data/train_lwir.txt"
    HEIGHT = 512
    WIDTH = 512
    lr = 1e-4 # "Initial learning rate, default is 0.0001"
    lr_step = 10 # Learning rate is halved in 10 epochs 	
    resume = "model/weights/ckpt_299.pt" # if you have, please put the path of the model like "./models/densefuse_gray.model"
    # resume = None
    save_model_dir = "./models/" #"path to folder where trained model with checkpoints will be saved."
    workers = 4

    # For GPU training
    world_size = -1
    rank = -1
    dist_backend = 'nccl'
    gpu = 0
    multiprocessing_distributed = False
    distributed = None

    # For testing
    test_save_dir = "data/fusimg/"
    # test_visible = "./data/test_visible.txt"
    # test_lwir = "./data/test_lwir.txt"
    # test_visible = "./data/train_visible.txt"
    # test_lwir = "./data/train_lwir.txt"