import torch
from torch_geometric.loader import DataLoader


def get_dataset(prog_args, num_samples = None, shuffle = True, normalize = False):
    if prog_args.dataset == "full_knn4":
        data = torch.load("/home/binal1/LD-FPG/blind/data/processed/full_data_knn_4.pt", weights_only = False)
    elif prog_args.dataset == "full_knn10":
        data = torch.load("/home/binal1/LD-FPG/blind/data/processed/full_data_knn_10.pt", weights_only = False)
        print(f"\n----------- Full atom knn = 10 -------------\n")
    elif prog_args.dataset == "full_knn25":
        data = torch.load("/home/binal1/LD-FPG/blind/data/processed/full_data_knn_25.pt", weights_only = False)
        print(f"\n----------- Full atom knn = 25 -------------\n")


    elif prog_args.dataset == "full_knn4_10":
        data = torch.load("data/full_data_knn_4_10.pt", weights_only = False)      
        print(f"\n----------- Full atom knn = 4 - interval 10 -------------\n")
    elif prog_args.dataset == "full_knn4_20":
        data = torch.load("data/full_data_knn_4_20.pt", weights_only = False)      
        print(f"\n----------- Full atom knn = 4 - interval 20 -------------\n")
    elif prog_args.dataset == "full_knn4_100":
        data = torch.load("data/full_data_knn_4_100.pt", weights_only = False)      
        print(f"\n----------- Full atom knn = 4 - interval 100 -------------\n")


    elif prog_args.dataset == "dyn_1119_heavy":
        data = torch.load("data/dyn_1119_knn_4_heavy.pt", weights_only = False)
        print(f"\n----------- Full atom dyn_1119 knn = 4 -  heavy atoms only -------------\n")

    elif prog_args.dataset == "dyn_95_heavy":
        data = torch.load("data/dyn_95_knn_4_heavy.pt", weights_only=False)
        print(f"\n----------- Backbone dyn_95 knn = 4 - heavy atoms only -------------\n")

    elif prog_args.dataset == "dyn_200_heavy":
        data = torch.load("data/dyn_200_knn_4_heavy.pt", weights_only=False)
        print(f"\n----------- Backbone dyn_200 knn = 4 - heavy atoms only -------------\n")



    
    if shuffle:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(data), generator=g).tolist()
        data = [data[i] for i in perm]
    
    if num_samples is not None:
        data = data[:num_samples]

    n_sample = len(data)
    n_train = round(n_sample*.9)
    print(f"n_sample = {n_sample}") 

    train_dataset = data[:n_train]
    test_dataset = data[n_train:]      


    n_card = 3    # number of coordinates 
    
    print(prog_args.dataset)
    print(f"Number of samples - train : {len(train_dataset)}")
    print(f"Number of samples - test : {len(test_dataset)}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=prog_args.batch_size, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=prog_args.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) # For evaluating and saving all embeddings


    return train_loader, test_loader, n_card

