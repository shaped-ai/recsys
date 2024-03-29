import torch
from pytorch_lightning.lite import LightningLite
from torch.utils.data import DataLoader
from tqdm import tqdm


class PytorchLightningLiteTrainer(LightningLite):
    def run(self):
        pass

    def fit(
        self,
        model,
        dataset,
        max_num_epochs=50,
        batch_size=512,
        lr_rate=0.01,
        early_stopping: int = None,
        **kwargs,
    ):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn_float32
        )
        dataloader = self.setup_dataloaders(dataloader)
        optimizer = model.configure_optimizers(lr_rate)
        model, optimizer = self.setup(model, optimizer)
        model.train()

        pbar = tqdm(range(max_num_epochs))
        epoch_losses = []
        best_loss = 1e10
        for epoch in pbar:
            epoch_loss = 0
            for idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model.training_step(batch)
                self.backward(loss)
                epoch_loss += loss.item()
                optimizer.step()
                if idx % 10 == 0:
                    pbar.set_description(
                        f"Epoch: {epoch}/{max_num_epochs}, Loss: {loss:.2f}"
                    )
            epoch_losses.append(epoch_loss)
            best_loss = min(best_loss, epoch_loss)
            # Check the last early_stopping N epochs values and if new value is lower than the previous one, stop training
            if early_stopping is not None and epoch > early_stopping:
                early_check = [
                    epoch_losses[-i] > best_loss for i in range(early_stopping)
                ]
                if all(early_check):
                    break

        print("Training finished")

    def evaluate(
        self, model, dataset, item_features, user_features, batch_size=512, **kwargs
    ):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader = self.setup_dataloaders(dataloader)
        model, _ = self.setup(model)
        model.eval()

        pbar = tqdm(range(len(dataloader)))
        for idx, batch in enumerate(dataloader):
            model.validation_step(batch)
            pbar.update()


def collate_fn_float32(batch_output):
    batch_output = torch.utils.data.dataloader.default_collate(batch_output)
    return batch_output
