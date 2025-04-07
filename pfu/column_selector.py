from abc import ABC, abstractmethod
from typing import List


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
