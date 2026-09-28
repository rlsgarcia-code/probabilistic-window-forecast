import json
import math

import pytest

from probabilistic_window_forecast import BayesianConfig, BayesianForecastPipeline


NOISE = [0.3, -0.4, 0.2, -0.1, 0.5, -0.2]


def test_probabilities_are_normalized_and_linear_no_change_is_recovered():
    series = [(100 + 2 * index + NOISE[index % 6], index) for index in range(30)]
    pipeline = BayesianForecastPipeline(
        BayesianConfig(min_before=8, min_after=8, posterior_draws=2_000)
    )

    result = pipeline.run(series)

    assert sum(item.posterior_probability for item in result.hypotheses) == pytest.approx(1)
    assert sum(result.model_probabilities.values()) == pytest.approx(1)
    assert result.probability_no_change + result.probability_change == pytest.approx(1)
    assert result.probability_no_change > 0.90
    assert result.recommended_model == "linear"


def test_strong_terminal_break_is_recovered():
    series = []
    for index in range(42):
        signal = (
            100 + 0.2 * index
            if index < 22
            else 118 + 2.4 * (index - 22)
        )
        series.append((signal + NOISE[index % 6], index))

    result = BayesianForecastPipeline(
        BayesianConfig(min_before=8, min_after=8, posterior_draws=2_000)
    ).run(series)

    assert result.probability_change > 0.99
    assert result.recommended_window_start_t == 22
    assert result.recommended_model == "linear"
    best = max(result.window_probabilities, key=lambda item: item.probability)
    assert best.start_index == 22
    assert best.probability > 0.85

    assert set(result.model_views) == set(result.model_probabilities)
    assert sum(
        view.conditional_probability for view in result.model_views.values()
    ) == pytest.approx(1)
    linear_view = result.model_views["linear"]
    assert linear_view.window_start_t == 22
    assert len(linear_view.fitted) == len(series) - 22
    assert len(linear_view.forecast) == 6


def test_forecast_variance_is_decomposed():
    series = [(50 + 0.5 * index + NOISE[index % 6], index) for index in range(24)]
    result = BayesianForecastPipeline(
        BayesianConfig(min_before=8, min_after=8, posterior_draws=2_000)
    ).run(series)

    for point in result.forecast:
        assert point.total_variance == pytest.approx(
            point.within_hypothesis_variance + point.between_hypothesis_variance
        )
        assert point.lower < point.median < point.upper

    assert result.predictive_uncertainty_index >= result.selection_uncertainty_index
    assert result.expected_decision_loss == pytest.approx(
        result.selection_uncertainty_index
    )


def test_manual_selection_preserves_automatic_inference():
    series = [(100 + 1.5 * index + NOISE[index % 6], index) for index in range(24)]
    pipeline = BayesianForecastPipeline(
        BayesianConfig(min_before=6, min_after=6, posterior_draws=2_000)
    )

    result = pipeline.run(series, force_no_change=True, model_override="constant")

    assert result.window_selection_source == "manual"
    assert result.model_selection_source == "manual"
    assert result.selected_no_change
    assert result.selected_model == "constant"
    assert result.recommended_model == "linear"
    assert result.intervention_resolved_by_user
    assert not result.human_intervention_required


def test_exponential_is_available_through_laplace():
    noise = [0.001, -0.002, 0.0015, -0.001, 0.002, -0.0015]
    series = [
        (100 * math.exp(0.05 * index) * (1 + noise[index % 6]), index)
        for index in range(20)
    ]
    result = BayesianForecastPipeline(
        BayesianConfig(
            min_before=20,
            min_after=20,
            candidate_models=("linear", "exponential"),
            center_series=False,
            posterior_draws=2_000,
            laplace_summary_draws=1_000,
        )
    ).run(series)

    exponential = next(item for item in result.hypotheses if item.model == "exponential")
    assert exponential.inference_method == "laplace"
    assert exponential.posterior_parameters["log_rate"] == pytest.approx(0.05, abs=0.01)
    assert result.model_probabilities["exponential"] > 0.90


def test_dict_input_batch_and_json_output():
    series = [{"valor": 30 + index, "t": f"2024-{index + 1:02d}"} for index in range(12)]
    pipeline = BayesianForecastPipeline(
        BayesianConfig(min_before=8, min_after=8, posterior_draws=2_000)
    )

    results = pipeline.run_many({"a": series, "b": series})
    payload = results["a"].to_dict()

    json.dumps(payload)
    assert set(results) == {"a", "b"}
    assert payload["series_id"] == "a"


def test_non_positive_series_excludes_exponential():
    series = [(-5 + index, index) for index in range(12)]
    result = BayesianForecastPipeline(
        BayesianConfig(
            min_before=8,
            min_after=8,
            candidate_models=("linear", "exponential"),
            posterior_draws=2_000,
        )
    ).run(series)

    assert set(result.model_probabilities) == {"linear"}


def test_robust_centering_and_unit_information_g_prior_are_reported():
    series = [(1_000 + 3 * index + NOISE[index % 6], index) for index in range(24)]
    result = BayesianForecastPipeline(
        BayesianConfig(min_before=24, min_after=24, posterior_draws=2_000)
    ).run(series)

    assert result.center == pytest.approx((series[11][0] + series[12][0]) / 2)
    assert result.g_value == pytest.approx(24.0)
    assert result.prior_strategy == "zellner_g_prior_unit_information"
    assert result.selected_design_condition_number is not None
    for hypothesis in result.hypotheses:
        assert hypothesis.g_value == pytest.approx(24.0)
        assert hypothesis.design_condition_number is not None


def test_g_sensitivity_is_exposed_and_serialized():
    series = [(80 + 1.2 * index + NOISE[index % 6], index) for index in range(24)]
    result = BayesianForecastPipeline(
        BayesianConfig(min_before=8, min_after=8, posterior_draws=2_000)
    ).run(series)

    assert {item.g_multiplier for item in result.g_sensitivity_results} == {
        0.5,
        1.0,
        2.0,
    }
    assert all(len(item.forecast_means) == 6 for item in result.g_sensitivity_results)
    assert result.max_g_sensitivity_forecast_spread >= 0
    payload = result.to_dict()
    assert "g_sensitivity_results" in payload
    assert "prior_sensitive" in payload


def test_exponential_requires_scale_only_mode_when_it_is_eligible():
    with pytest.raises(ValueError, match="center_series=False"):
        BayesianForecastPipeline(
            BayesianConfig(
                min_before=8,
                min_after=8,
                candidate_models=("linear", "exponential"),
                posterior_draws=2_000,
            )
        ).run([(100 + index, index) for index in range(16)])
