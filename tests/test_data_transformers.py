import pytest
from pfu.data_transformers import NothingTransformer, TargetRollingAverages

@pytest.mark.parametrize("transformer", [NothingTransformer(), TargetRollingAverages()])
def test_transformers(transformer):
    assert 1 == 1
