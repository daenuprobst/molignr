import torch
from torch_geometric.loader import DataLoader

def get_dataset(prog_args, num_samples = None, shuffle = True):

    # this dataset contains only the first 200 samples in the trajectory
    data = torch.load("dataset.pt", weights_only = False)

    # If you want to experiment with the full trajectory, uncomment the one below
    # data = torch.load("full_atom_structure_knn_4.pt", weights_only = False)

    if num_samples is not None:
        data = data[:num_samples]

    n_sample = len(data)
    n_train = round(n_sample*.9)
    print(f"n_sample = {n_sample}") 

    train_dataset = data[:n_train]
    test_dataset = data[n_train:]  

    n_card = 3    
    
    print(prog_args.dataset)
    print(f"Number of samples - train : {len(train_dataset)}")
    print(f"Number of samples - test : {len(test_dataset)}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=prog_args.batch_size, shuffle=shuffle, drop_last=True, num_workers=4, pin_memory=True) # Data is pre-shuffled by fixed seed
    test_loader  = DataLoader(test_dataset, batch_size=prog_args.batch_size, shuffle=False, drop_last=True, num_workers=4, pin_memory=True) # For evaluating and saving all embeddings

    return train_loader, test_loader, n_card

