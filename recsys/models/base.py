from abc import ABC, abstractmethod


class BaseModel(ABC):
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

    def compile(
        self,
    ):
        pass

    @property
    def trainer(self):
        return self._trainer

    def fit(self, **kwargs):
        self.trainer.fit(model=self, **kwargs)


#     @abstractmethod
#     def get_n_recommendation_batch(self, query_vectors, n, params):
#         """
#         Recommendation for batch users in list.
#         """
#         pass
