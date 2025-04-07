from abc import ABC, abstractmethod
from typing import List
from pfu.data_transformers import (
    col_prefix_target,
    col_prefix_past_covariate,
    col_prefix_future_covariate,
    col_prefix_static_covariate,
)


class ColumnSelector(ABC):
    @abstractmethod
    def column_is_selected(self, column_name: str):
        pass

    def column_subset(self, columns: List[str]):
        return [c for c in columns if self.column_is_selected(c)]

    @abstractmethod
    def get_params(self):
        pass

    def get_object(self):
        return {"object": self.__class__.__name__, "params": self.get_params()}


class CovariateSelector(ColumnSelector):
    def __init__(
        self,
        select_target=False,
        select_past_covariates=False,
        select_future_covariates=False,
        select_static_covariates=False,
    ):
        super().__init__()
        self.select_past_covariates = select_past_covariates
        self.select_future_covariates = select_future_covariates
        self.select_static_covariates = select_static_covariates
        self.select_target = select_target

    def column_is_selected(self, column_name: str):
        if column_name.startswith(f"{col_prefix_past_covariate}|") and self.select_past_covariates:
            return True
        if column_name.startswith(f"{col_prefix_future_covariate}|") and self.select_future_covariates:
            return True
        if column_name.startswith(f"{col_prefix_static_covariate}|") and self.select_static_covariates:
            return True
        if column_name.startswith(f"{col_prefix_target}|") and self.select_target:
            return True
        return False

    def get_params(self):
        return {
            "select_past_covariates": self.select_past_covariates,
            "select_future_covariates": self.select_future_covariates,
            "select_static_covariates": self.select_static_covariates,
            "select_target": self.select_target,
        }


class AllColumnsSelector(CovariateSelector):
    def __init__(self):
        super().__init__(
            select_target=True,
            select_past_covariates=True,
            select_future_covariates=True,
            select_static_covariates=True,
        )


class TargetColumnsSelector(CovariateSelector):
    def __init__(self):
        super().__init__(
            select_target=True,
            select_past_covariates=False,
            select_future_covariates=False,
            select_static_covariates=False,
        )


class PastCovariateColumnsSelector(CovariateSelector):
    def __init__(self):
        super().__init__(
            select_target=False,
            select_past_covariates=True,
            select_future_covariates=False,
            select_static_covariates=False,
        )
