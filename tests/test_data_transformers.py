from pfu.utils import generate_synthetic_data
import pytest
from pfu.data_transformers import DropCovariatesTransformer, NothingTransformer, TargetRollingAverages


@pytest.mark.parametrize("transformer_class", [NothingTransformer, TargetRollingAverages, DropCovariatesTransformer])
def test_transformers(transformer_class):
    s = generate_synthetic_data()
    t = transformer_class()
    t.fit_transform(s)
    assert 1 == 1
