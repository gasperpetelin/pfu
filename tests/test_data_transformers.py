import pytest
from pfu.data_transformers import NothingTransformer, TargetRollingAverages


@pytest.mark.parametrize("transformer_class", [NothingTransformer, TargetRollingAverages])
def test_transformers(transformer_class):
    t = transformer_class()
    assert 1 == 1
