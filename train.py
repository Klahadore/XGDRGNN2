import torch
import pickle
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import norm


from model import Model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":


    
    model = Model(384, training=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    data_loader = LinkNeighborLoader(
        new_train_dataset,
        num_neighbors=[32, 24],
        batch_size=384,
        shuffle=True,
        edge_label=new_train_dataset.edge_label[
            : new_train_dataset.edge_label_index.shape[1]
        ],
        edge_label_index=new_train_dataset.edge_label_index,
        num_workers=4,
    )

    scaler = torch.cuda.amp.GradScaler()

    def train(loader):
        model.train()
        total_loss, total_examples = 0, 0
        for batch_data in loader:
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            pred, _ = model(batch_data)
            target = batch_data.edge_label[: len(pred)]
            print(pred)
            print(target)
            loss = nn.BCEWithLogitsLoss()(pred, target.float())

            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            #  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            print(loss.item())

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        return total_loss / total_examples

    for epoch in range(8):
        epoch_loss = train(data_loader)
        print(f"Epoch {epoch}: Loss {epoch_loss}")
        if epoch == 1:
            torch.save(model.state_dict(), "cheese_epoch2_3.pt")
            print("epoch 2 saved")
        if epoch == 3:
            torch.save(model.state_dict(), "cheese_epoch4_3.pt")
            print("epoch 4 saved")
        if epoch == 5:
            torch.save(model.state_dict(), "cheese_epoch6_3.pt")
            print("epoch 6 saved")
        if epoch == 7:
            torch.save(model.state_dict(), "cheese_epoch8_3.pt")
            print("epoch 8 saved")

    print("done")
