from abc import ABC, abstractmethod
import os
import time
import polars as pl
from typing import List, Union
import pickle


class Transformer(ABC):
    def __init__(self, verbose=True, requires_fitting=True):
        self.col_id = None
        self.col_target = None
        self.col_timestamp = None
        self.col_past_covariates = None
        self.col_future_covariates = None
        self.col_static_covariates = None
        self.verbose = verbose
        self.requires_fitting = requires_fitting
        if self.requires_fitting:
            self._is_fitted = False
        else:
            self._is_fitted = True
        self._columns_inferred = False

    def fit(self, lf: pl.LazyFrame):
        self.infer_meta_columns_if_not_already_done(lf)
        start_time = time.time()
        self._fit(lf)
        end_time = time.time()
        self._is_fitted = True
        if self.verbose:
            print(
                f"{self.__class__.__name__} fit: (None, {len(lf.collect_schema().names())}) -> (None, {len(lf.collect_schema().names())}) in {end_time - start_time:.2f} seconds"
            )

    def transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        self.infer_meta_columns_if_not_already_done(lf)
        assert isinstance(lf, pl.LazyFrame), "Input must be a polars LazyFrame."
        assert self._is_fitted, f"Transformer {self.__class__.__name__} must be fitted before transforming."

        input_columns_len = len(lf.collect_schema().names())
        start_time = time.time()
        new_lf = self._transform(lf)
        end_time = time.time()
        output_columns_len = len(new_lf.collect_schema().names())
        output_row_len = None
        if isinstance(new_lf, pl.DataFrame):
            output_row_len = new_lf.height
        if self.verbose:
            print(
                f"{self.__class__.__name__} transform: (None, {input_columns_len}) -> ({output_row_len}, {output_columns_len}) in {end_time - start_time:.2f} seconds"
            )
        return new_lf.lazy()

    def rename_target_prefixes_to_past_covariates(self, columns: List[str]):
        return [c.replace(f"{col_prefix_target}|", f"{col_prefix_past_covariate}|") for c in columns]

    def _fit(self, lf: pl.LazyFrame) -> None:
        pass

    @abstractmethod
    def _transform(self, lf: pl.LazyFrame) -> Union[pl.DataFrame, pl.LazyFrame]:
        pass

    def fit_transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        self.fit(lf)
        return self.transform(lf)

    def meta_columns(self) -> List[str]:
        return [self.col_timestamp, self.col_id, self.col_target]

    def infer_meta_columns_if_not_already_done(self, lf):
        if self._columns_inferred == False:
            (
                self.col_id,
                self.col_timestamp,
                self.col_target,
                self.col_past_covariates,
                self.col_future_covariates,
                self.col_static_covariates,
            ) = extract_inferred_meta_columns(lf=lf)
            self._columns_inferred = True

    @abstractmethod
    def get_params(self) -> dict:
        pass

    def get_object(self):
        return {"object": self.__class__.__name__, "params": self.get_params()}

    def save(self, directory: str):
        print(f"Saving transformer to {directory}")
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, "component.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(directory: str):
        file_path = os.path.join(directory, "component.pkl")
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        param_string = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])
        return f"{self.__class__.__name__}({param_string})"

    def __str__(self) -> str:
        return self.__repr__()


class NothingTransformer(Transformer):
    def get_params(self):
        return {}

    def _transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf


class DateTimeFeatures(Transformer):
    def __init__(self, verbose=True):
        super().__init__(verbose=verbose, requires_fitting=False)

    def get_params(self):
        return {}

    def _transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        tc = self.col_timestamp
        tc = "|".join(tc.split("|")[1:])
        print(tc)
        return lf.with_columns(
            pl.col(self.col_timestamp).dt.month().over(self.col_id).alias(f"{col_prefix_future_covariate}|{tc}|month"),
            pl.col(self.col_timestamp).dt.day().over(self.col_id).alias(f"{col_prefix_future_covariate}|{tc}|day"),
            pl.col(self.col_timestamp).dt.hour().over(self.col_id).alias(f"{col_prefix_future_covariate}|{tc}|hour"),
            pl.col(self.col_timestamp)
            .dt.weekday()
            .over(self.col_id)
            .alias(f"{col_prefix_future_covariate}|{tc}|weekday"),
            pl.col(self.col_timestamp).dt.week().over(self.col_id).alias(f"{col_prefix_future_covariate}|{tc}|week"),
            pl.col(self.col_timestamp).dt.year().over(self.col_id).alias(f"{col_prefix_future_covariate}|{tc}|year"),
        )


class TargetRollingAverages(Transformer):
    def __init__(self, windows: List[int] = [3, 5, 11, 21, 47]):
        super().__init__(requires_fitting=False)
        self.windows = windows

    def _transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        for ws in self.windows:
            c_alias = f"{self.col_target}|rolling_mean;window_size={ws}".replace(
                f"{col_prefix_target}|", f"{col_prefix_past_covariate}|"
            )
            lf = lf.with_columns(pl.col(self.col_target).rolling_mean(window_size=ws).over(self.col_id).alias(c_alias))
        return lf

    def get_params(self):
        return {"windows": self.windows}


class DropCovariatesTransformer(Transformer):
    def _transform(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select(self.meta_columns())

    def get_params(self):
        return {}
