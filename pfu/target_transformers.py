from typing import Union
from pfu import data_transformers
import polars as pl
from pfu import utils


class LogPCTargetTransformer(data_transformers.Transformer):
    def __init__(self, constant: float = 0.1, verbose: bool = True):
        super().__init__(verbose=verbose, requires_fitting=False)
        self.constant = constant

    def _transform(self, lf: pl.LazyFrame) -> Union[pl.DataFrame, pl.LazyFrame]:
        lf = lf.with_columns(
            (pl.col(self.col_target) + self.constant)
            .log()
            .alias(f"{self.col_target}{utils.col_delimiter}log;c={self.constant}")
        ).drop(self.col_target)
        return lf

    def inverse_transform(self, lf: pl.LazyFrame) -> Union[pl.DataFrame, pl.LazyFrame]:
        lf = lf.with_columns(
            (pl.col(f"{self.col_target}{utils.col_delimiter}log;c={self.constant}").exp() - self.constant).alias(
                self.col_target
            )
        ).drop(f"{self.col_target}{utils.col_delimiter}log;c={self.constant}")
        return lf

    def get_params(self):
        return {"constant": self.constant}
