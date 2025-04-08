import numpy as np
import polars as pl

col_prefix_id = "id"
col_prefix_target = "target"
col_prefix_timestamp = "timestamp"
col_prefix_past_covariate = "covpast"
col_prefix_future_covariate = "covfuture"
col_prefix_static_covariate = "covstatic"
col_prefix_prediction = "pred"
col_delimiter = "|"


def extract_inferred_meta_columns(lf):
    columns = lf.collect_schema().keys()

    col_ids = [c for c in columns if c.startswith(f"{col_prefix_id}{col_delimiter}")]
    col_targets = [c for c in columns if c.startswith(f"{col_prefix_target}{col_delimiter}")]
    col_timestamps = [c for c in columns if c.startswith(f"{col_prefix_timestamp}{col_delimiter}")]
    col_past_covariates = [c for c in columns if c.startswith(f"{col_prefix_past_covariate}{col_delimiter}")]
    col_future_covariates = [c for c in columns if c.startswith(f"{col_prefix_future_covariate}{col_delimiter}")]
    col_static_covariates = [c for c in columns if c.startswith(f"{col_prefix_static_covariate}{col_delimiter}")]

    assert len(col_ids) == 1, "There must be exactly one ID column."
    assert len(col_timestamps) == 1, "There must be exactly one timestamp column."

    col_id = col_ids[0]
    col_timestamp = col_timestamps[0]
    col_target = col_targets[0] if col_targets else None

    return (
        col_id,
        col_timestamp,
        col_target,
        col_past_covariates,
        col_future_covariates,
        col_static_covariates,
    )


def generate_synthetic_data(
    n_series=3,
    frequency="1m",
    n_correlated_past_covariates=2,
    n_random_noise_past_covariates=3,
    n_zero_variance_past_covariates=2,
    seed=42,
    apply_random_translation=False,
    apply_random_scale=False,
    apply_random_streach=False,
):
    timestamp = pl.datetime_range(
        start=pl.datetime(2023, 1, 1), end=pl.datetime(2024, 1, 1), interval=frequency, eager=True
    )
    series_len = len(timestamp)
    data = []

    np.random.seed(seed)
    for series_id in range(n_series):
        translation_factor = np.random.normal(0, 10) if apply_random_translation else 0
        scale_factor = min(np.random.exponential(), 5) if apply_random_scale else 1
        streach_factor = max(0.5, min(np.random.exponential(), 2)) if apply_random_streach else 1

        mid_price = np.sin(
            np.linspace(series_id, series_len / 100 + series_id, series_len) * streach_factor
        ) + np.random.normal(0, 0.05, series_len)
        mid_price = mid_price * scale_factor + translation_factor

        correlated_covariates = [
            mid_price + np.random.normal(0, 0.2, series_len) for _ in range(n_correlated_past_covariates)
        ]
        random_noise_covariates = [np.random.normal(0, 1, series_len) for _ in range(n_random_noise_past_covariates)]

        series_data = {
            f"{col_prefix_id}{col_delimiter}unique_id": [f"id_{series_id}"] * series_len,
            f"{col_prefix_timestamp}{col_delimiter}timestamp": timestamp,
            f"{col_prefix_target}{col_delimiter}value": mid_price,
        }

        for i, covariate in enumerate(correlated_covariates, 1):
            series_data[f"{col_prefix_past_covariate}{col_delimiter}correlated_cov_{i}"] = covariate

        for i, noise_covariate in enumerate(random_noise_covariates, 1):
            series_data[f"{col_prefix_future_covariate}{col_delimiter}noise_cov_{i}"] = noise_covariate

        for i in range(n_zero_variance_past_covariates):
            series_data[f"{col_prefix_static_covariate}{col_delimiter}zero_variance_cov_{i+1}"] = np.ones(series_len)

        data.append(pl.DataFrame(series_data))

    np.random.seed(None)
    return pl.concat(data).lazy()
