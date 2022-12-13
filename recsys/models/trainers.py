from pytorch_lightning.lite import LightningLite
from torch.utils.data import DataLoader
from tqdm import tqdm


class PytorchLightningLiteTrainer(LightningLite):
    def run(self):
        pass

    def fit(self, model, dataset, num_epochs=50, batch_size=512, **kwargs):
        dataloader = DataLoader(dataset, batch_size=batch_size)
        dataloader = self.setup_dataloaders(dataloader)
        optimizer = model.configure_optimizers()
        model, optimizer = self.setup(model, optimizer)
        model.train()

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            for idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model.training_step(batch)
                self.backward(loss)
                optimizer.step()
                if idx % 10 == 0:
                    pbar.set_description(
                        f"Epoch: {epoch}/{num_epochs}, Loss: {loss:.2f}"
                    )

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
