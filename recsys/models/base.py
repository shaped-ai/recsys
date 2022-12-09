from abc import ABC, abstractmethod


class BaseModel(ABC):
    # PL Related methods
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("`forward` method must be implemented by the user")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError(
            "`training_step` method must be implemented by the user"
        )

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError(
            "`validation_step` method must be implemented by the user"
        )

    def configure_optimizers(self):
        raise NotImplementedError(
            "`configure_optimizers` method must be implemented by the user"
        )

    def fit(self, dataset, **kwargs):
        self._trainer.fit(model=self, dataset=dataset, **kwargs)

    def batch_score(self, args):
        return NotImplementedError("`batch_score` method must be implemented by the user")
    def encode_user(self, user):
        raise NotImplementedError("`encode_user` method must be implemented by the user")

    def encode_item(self, item):
        raise NotImplementedError("`encode_item` method must be implemented by the user")
