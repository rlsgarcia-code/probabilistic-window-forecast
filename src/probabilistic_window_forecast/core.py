"""Inferencia conjunta de janela, modelo e forecast.

Os modelos constante, linear e logaritmico usam regressao bayesiana
Normal-Inverse-Gamma conjugada, centralizacao robusta e g-prior de Zellner. O
exponencial e opcional e usa aproximacao de Laplace. Todas as hipoteses
explicam o mesmo lookback completo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import calendar
import math
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp


SUPPORTED_MODELS = ("constant", "linear", "logarithmic", "exponential")


@dataclass(frozen=True)
class Observation:
    """Uma observacao; tuplas de entrada usam a ordem ``(valor, t)``."""

    value: float
    t: Any


@dataclass
class BayesianConfig:
    max_lookback: int = 60
    min_before: int = 8
    min_after: int = 8
    forecast_horizon: int = 6
    confidence: float = 0.95
    credible_mass: float = 0.90
    candidate_models: tuple[str, ...] = (
        "constant",
        "linear",
        "logarithmic",
    )
    prior_no_change: float = 0.60
    model_priors: Mapping[str, float] | None = None
    center_series: bool = True
    g_prior_multiplier: float = 1.0
    g_sensitivity_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)
    max_design_condition_number: float = 1e8
    # Mantidos apenas para o modelo exponencial experimental.
    prior_intercept_variance: float = 100.0
    prior_slope_variance: float = 1.0
    prior_log_amplitude_variance: float = 4.0
    prior_exponential_rate_variance: float = 0.04
    noise_shape: float = 2.0
    noise_scale: float = 0.001
    max_abs_exponential_rate: float = 0.50
    posterior_draws: int = 12_000
    laplace_summary_draws: int = 3_000
    random_seed: int = 1729
    materiality: float = 1.0
    review_cost: float = 0.05

    def __post_init__(self) -> None:
        if self.max_lookback < 4:
            raise ValueError("max_lookback deve ser pelo menos 4")
        if self.min_before < 2 or self.min_after < 2:
            raise ValueError("min_before e min_after devem ser pelo menos 2")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon deve ser positivo")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence deve estar entre zero e um")
        if not 0 < self.credible_mass < 1:
            raise ValueError("credible_mass deve estar entre zero e um")
        if not 0 < self.prior_no_change < 1:
            raise ValueError("prior_no_change deve estar entre zero e um")
        if self.posterior_draws < 1_000:
            raise ValueError("posterior_draws deve ser pelo menos 1000")
        if self.noise_shape <= 0 or self.noise_scale <= 0:
            raise ValueError("parametros da prior de ruido devem ser positivos")
        unknown = set(self.candidate_models) - set(SUPPORTED_MODELS)
        if unknown:
            raise ValueError(f"modelos nao suportados: {sorted(unknown)}")
        if not self.candidate_models:
            raise ValueError("candidate_models nao pode ser vazio")
        if self.g_prior_multiplier <= 0:
            raise ValueError("g_prior_multiplier deve ser positivo")
        if not self.g_sensitivity_multipliers or any(
            value <= 0 for value in self.g_sensitivity_multipliers
        ):
            raise ValueError("g_sensitivity_multipliers deve conter valores positivos")
        if self.max_design_condition_number <= 1:
            raise ValueError("max_design_condition_number deve ser maior que um")


@dataclass(frozen=True)
class HypothesisResult:
    model: str
    changed: bool
    start_index: int | None
    start_t: Any | None
    log_evidence: float
    log_prior: float
    posterior_probability: float
    inference_method: str
    posterior_parameters: dict[str, float]
    posterior_noise_variance: float
    g_value: float | None
    design_condition_number: float | None


@dataclass(frozen=True)
class GSensitivityResult:
    g_multiplier: float
    g_value: float
    recommended_window_start_t: Any | None
    recommended_no_change: bool
    recommended_model: str
    probability_no_change: float
    model_probabilities: dict[str, float]
    forecast_means: tuple[float, ...]


@dataclass(frozen=True)
class WindowProbability:
    changed: bool
    start_index: int | None
    start_t: Any | None
    probability: float


@dataclass(frozen=True)
class PredictivePoint:
    horizon: int
    t: Any | None
    mean: float
    median: float
    lower: float
    upper: float
    within_hypothesis_variance: float
    between_hypothesis_variance: float
    total_variance: float


@dataclass(frozen=True)
class FittedPoint:
    t: Any
    value: float


@dataclass(frozen=True)
class ModelView:
    model: str
    window_start_t: Any | None
    no_change: bool
    conditional_probability: float
    fitted: tuple[FittedPoint, ...]
    forecast: tuple[PredictivePoint, ...]


@dataclass(frozen=True)
class BayesianResult:
    series_id: str | None
    observation_count: int
    lookback_observations: tuple[Observation, ...]
    center: float
    scale: float
    prior_strategy: str
    g_value: float
    g_sensitivity_results: tuple[GSensitivityResult, ...]
    prior_sensitive: bool
    max_g_sensitivity_forecast_spread: float
    selected_design_condition_number: float | None
    maximum_design_condition_number: float | None
    probability_no_change: float
    probability_change: float
    window_probabilities: tuple[WindowProbability, ...]
    window_credible_set: tuple[Any | None, ...]
    model_probabilities: dict[str, float]
    hypotheses: tuple[HypothesisResult, ...]
    recommended_window_start_t: Any | None
    recommended_no_change: bool
    recommended_model: str
    selected_window_start_t: Any | None
    selected_no_change: bool
    window_selection_source: str
    selected_model: str
    model_selection_source: str
    forecast: tuple[PredictivePoint, ...]
    selected_forecast: tuple[PredictivePoint, ...]
    model_forecasts: dict[str, tuple[PredictivePoint, ...]]
    model_views: dict[str, ModelView]
    predictive_uncertainty_index: float
    selection_uncertainty_index: float
    expected_decision_loss: float
    review_cost: float
    human_intervention_required: bool
    intervention_resolved_by_user: bool

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class _HypothesisState:
    model: str
    changed: bool
    start_index: int | None
    start_t: Any | None
    log_evidence: float
    log_prior: float
    method: str
    center: float
    scale: float
    parameter_names: tuple[str, ...]
    posterior_mean: np.ndarray
    posterior_noise_variance: float
    g_value: float | None = None
    design_condition_number: float | None = None
    probability: float = 0.0
    # Campos conjugados
    vn: np.ndarray | None = None
    an: float | None = None
    dn: float | None = None
    # Campos de Laplace; inclui log(sigma^2) como ultimo parametro.
    laplace_covariance: np.ndarray | None = None


class BayesianForecastPipeline:
    """Pipeline probabilistico para uma ou muitas series."""

    def __init__(self, config: BayesianConfig | None = None) -> None:
        self.config = config or BayesianConfig()

    def run(
        self,
        observations: Iterable[Observation | Mapping[str, Any] | Sequence[Any]],
        *,
        series_id: str | None = None,
        window_start: Any | None = None,
        force_no_change: bool = False,
        model_override: str | None = None,
        materiality: float | None = None,
    ) -> BayesianResult:
        if window_start is not None and force_no_change:
            raise ValueError("window_start e force_no_change nao podem ser usados juntos")

        parsed = tuple(_parse_observation(item) for item in observations)
        self._validate(parsed)
        lookback = parsed[-self.config.max_lookback :]
        y = np.asarray([item.value for item in lookback], dtype=float)
        center = float(np.median(y)) if self.config.center_series else 0.0
        centered = y - center
        mad_scale = 1.4826 * float(np.median(np.abs(centered)))
        if self.config.center_series:
            scale = (
                mad_scale
                if mad_scale > 1e-8
                else max(float(np.std(centered)), abs(center), 1.0)
            )
        else:
            scale = max(float(np.median(np.abs(y))), float(np.std(y)), 1e-8)
        z = centered / scale

        eligible_models = self._eligible_models(y)
        model_priors = self._normalized_model_priors(eligible_models)
        states = self._enumerate_hypotheses(
            lookback,
            z,
            center,
            scale,
            model_priors,
            g_multiplier=self.config.g_prior_multiplier,
        )
        self._normalize_hypotheses(states)

        hypotheses = tuple(self._public_hypothesis(state) for state in states)
        window_probabilities = self._window_probabilities(states, lookback)
        model_probabilities = self._model_probabilities(states)
        probability_no_change = sum(state.probability for state in states if not state.changed)
        probability_change = 1.0 - probability_no_change

        best_window = max(window_probabilities, key=lambda item: item.probability)
        recommended_model = max(model_probabilities, key=model_probabilities.get)
        credible_set = self._credible_set(window_probabilities)

        sensitivity_results = self._g_sensitivity(
            lookback,
            z,
            center,
            scale,
            model_priors,
            base_states=states,
        )
        sensitivity_selections = {
            (
                item.recommended_no_change,
                item.recommended_window_start_t,
                item.recommended_model,
            )
            for item in sensitivity_results
        }
        prior_sensitive = len(sensitivity_selections) > 1
        sensitivity_means = np.asarray(
            [item.forecast_means for item in sensitivity_results], dtype=float
        )
        max_g_spread = float(np.max(np.ptp(sensitivity_means, axis=0)) / scale)

        rng = np.random.default_rng(self.config.random_seed)
        forecast = self._mixture_forecast(states, lookback, rng)
        model_forecasts = {
            model: self._mixture_forecast(
                [state for state in states if state.model == model],
                lookback,
                np.random.default_rng(self.config.random_seed + index + 1),
                renormalize=True,
            )
            for index, model in enumerate(eligible_models)
        }

        selected_states, selected_window_t, selected_no_change, window_source = (
            self._selected_window_states(
                states,
                lookback,
                best_window,
                window_start=window_start,
                force_no_change=force_no_change,
            )
        )
        selected_window_states = list(selected_states)
        selected_window_probability = sum(
            state.probability for state in selected_window_states
        )
        model_views = {
            state.model: ModelView(
                model=state.model,
                window_start_t=selected_window_t,
                no_change=selected_no_change,
                conditional_probability=float(
                    state.probability / selected_window_probability
                ),
                fitted=self._current_fitted(state, lookback),
                forecast=self._mixture_forecast(
                    [state],
                    lookback,
                    np.random.default_rng(
                        self.config.random_seed + 20_000 + index
                    ),
                    renormalize=True,
                ),
            )
            for index, state in enumerate(selected_window_states)
        }

        selected_model = recommended_model
        model_source = "automatic"
        if model_override is not None:
            if model_override not in eligible_models:
                raise ValueError(
                    f"model_override invalido; modelos elegiveis: {eligible_models}"
                )
            selected_model = model_override
            model_source = "manual"
        selected_states = [
            state for state in selected_states if state.model == selected_model
        ]
        if not selected_states:
            raise ValueError("a combinacao de janela e modelo selecionada nao e elegivel")
        selected_forecast = self._mixture_forecast(
            selected_states,
            lookback,
            np.random.default_rng(self.config.random_seed + 10_000),
            renormalize=True,
        )

        effective_materiality = (
            self.config.materiality if materiality is None else float(materiality)
        )
        normalized_total = np.mean(
            [point.total_variance / (scale**2) for point in forecast]
        )
        normalized_between = np.mean(
            [point.between_hypothesis_variance / (scale**2) for point in forecast]
        )
        expected_loss = float(effective_materiality * normalized_between)
        intervention_resolved = (
            window_start is not None or force_no_change or model_override is not None
        )
        human_intervention_required = (
            (expected_loss > self.config.review_cost or prior_sensitive)
            and not intervention_resolved
        )
        selected_condition = selected_states[0].design_condition_number
        finite_conditions = [
            state.design_condition_number
            for state in states
            if state.design_condition_number is not None
        ]
        maximum_condition = max(finite_conditions) if finite_conditions else None
        prior_strategy = "zellner_g_prior_unit_information"
        if "exponential" in eligible_models:
            prior_strategy += "_with_experimental_laplace_prior"

        return BayesianResult(
            series_id=series_id,
            observation_count=len(parsed),
            lookback_observations=lookback,
            center=center,
            scale=scale,
            prior_strategy=prior_strategy,
            g_value=float(len(z) * self.config.g_prior_multiplier),
            g_sensitivity_results=sensitivity_results,
            prior_sensitive=prior_sensitive,
            max_g_sensitivity_forecast_spread=max_g_spread,
            selected_design_condition_number=selected_condition,
            maximum_design_condition_number=maximum_condition,
            probability_no_change=float(probability_no_change),
            probability_change=float(probability_change),
            window_probabilities=window_probabilities,
            window_credible_set=credible_set,
            model_probabilities=model_probabilities,
            hypotheses=hypotheses,
            recommended_window_start_t=best_window.start_t,
            recommended_no_change=not best_window.changed,
            recommended_model=recommended_model,
            selected_window_start_t=selected_window_t,
            selected_no_change=selected_no_change,
            window_selection_source=window_source,
            selected_model=selected_model,
            model_selection_source=model_source,
            forecast=forecast,
            selected_forecast=selected_forecast,
            model_forecasts=model_forecasts,
            model_views=model_views,
            predictive_uncertainty_index=float(normalized_total),
            selection_uncertainty_index=float(normalized_between),
            expected_decision_loss=expected_loss,
            review_cost=self.config.review_cost,
            human_intervention_required=human_intervention_required,
            intervention_resolved_by_user=intervention_resolved,
        )

    def run_many(
        self,
        series: Mapping[
            str,
            Iterable[Observation | Mapping[str, Any] | Sequence[Any]],
        ],
    ) -> dict[str, BayesianResult]:
        return {
            series_id: self.run(values, series_id=series_id)
            for series_id, values in series.items()
        }

    def _validate(self, observations: tuple[Observation, ...]) -> None:
        if len(observations) < 4:
            raise ValueError("sao necessarias pelo menos 4 observacoes")
        for index, item in enumerate(observations):
            if not math.isfinite(item.value):
                raise ValueError(f"valor nao finito na posicao {index}")
        if any(
            observations[index].t == observations[index - 1].t
            for index in range(1, len(observations))
        ):
            raise ValueError("rotulos t consecutivos nao podem ser duplicados")

    def _eligible_models(self, y: np.ndarray) -> tuple[str, ...]:
        models = tuple(
            model
            for model in self.config.candidate_models
            if model != "exponential" or np.all(y > 0)
        )
        if "exponential" in models and self.config.center_series:
            raise ValueError(
                "o exponencial experimental exige center_series=False; "
                "a centralizacao robusta e usada pelos modelos conjugados"
            )
        return models

    def _normalized_model_priors(self, models: tuple[str, ...]) -> dict[str, float]:
        if not models:
            raise ValueError("nenhum modelo elegivel")
        supplied = self.config.model_priors or {model: 1.0 for model in models}
        values = {model: float(supplied.get(model, 0.0)) for model in models}
        if any(value <= 0 for value in values.values()):
            raise ValueError("model_priors deve conter pesos positivos para modelos elegiveis")
        total = sum(values.values())
        return {model: value / total for model, value in values.items()}

    def _enumerate_hypotheses(
        self,
        observations: tuple[Observation, ...],
        z: np.ndarray,
        center: float,
        scale: float,
        model_priors: dict[str, float],
        *,
        g_multiplier: float,
    ) -> list[_HypothesisState]:
        n = len(z)
        starts = list(range(self.config.min_before, n - self.config.min_after + 1))
        has_change_candidates = bool(starts)
        states: list[_HypothesisState] = []

        for model, model_prior in model_priors.items():
            no_change_prior = model_prior * (
                self.config.prior_no_change if has_change_candidates else 1.0
            )
            states.append(
                self._fit_hypothesis(
                    z,
                    center,
                    scale,
                    model=model,
                    changed=False,
                    start_index=None,
                    start_t=None,
                    log_prior=math.log(no_change_prior),
                    g_multiplier=g_multiplier,
                )
            )
            if has_change_candidates:
                each_change_prior = (
                    model_prior
                    * (1.0 - self.config.prior_no_change)
                    / len(starts)
                )
                for start in starts:
                    states.append(
                        self._fit_hypothesis(
                            z,
                            center,
                            scale,
                            model=model,
                            changed=True,
                            start_index=start,
                            start_t=observations[start].t,
                            log_prior=math.log(each_change_prior),
                            g_multiplier=g_multiplier,
                        )
                    )
        return states

    def _fit_hypothesis(
        self,
        z: np.ndarray,
        center: float,
        scale: float,
        *,
        model: str,
        changed: bool,
        start_index: int | None,
        start_t: Any | None,
        log_prior: float,
        g_multiplier: float,
    ) -> _HypothesisState:
        if model == "exponential":
            return self._fit_exponential_laplace(
                z,
                center,
                scale,
                changed=changed,
                start_index=start_index,
                start_t=start_t,
                log_prior=log_prior,
            )

        design, names = self._design_matrix(
            len(z), model=model, changed=changed, start_index=start_index
        )
        b0 = np.zeros(design.shape[1])
        condition_number = float(np.linalg.cond(design))
        if (
            not math.isfinite(condition_number)
            or condition_number > self.config.max_design_condition_number
        ):
            raise RuntimeError(
                f"matriz de desenho mal condicionada: {condition_number:.3g}"
            )
        data_precision = design.T @ design
        g_value = float(len(z) * g_multiplier)
        # Prior de Zellner: V0 = g (X'X)^-1. Logo, a informacao
        # anterior e X'X/g: o equivalente a 1/g da informacao dos dados.
        v0 = g_value * np.linalg.inv(data_precision)
        v0_inv = data_precision / g_value
        vn = np.linalg.inv(v0_inv + data_precision)
        bn = vn @ (v0_inv @ b0 + design.T @ z)
        an = self.config.noise_shape + len(z) / 2.0
        dn = self.config.noise_scale + 0.5 * float(
            z @ z + b0 @ v0_inv @ b0 - bn @ np.linalg.solve(vn, bn)
        )
        sign0, logdet0 = np.linalg.slogdet(v0)
        signn, logdetn = np.linalg.slogdet(vn)
        if sign0 <= 0 or signn <= 0 or dn <= 0:
            raise RuntimeError("posterior conjugada numericamente invalida")
        log_evidence = (
            -0.5 * len(z) * math.log(2.0 * math.pi)
            + 0.5 * (logdetn - logdet0)
            + self.config.noise_shape * math.log(self.config.noise_scale)
            - an * math.log(dn)
            + gammaln(an)
            - gammaln(self.config.noise_shape)
        )
        noise_mean = dn / (an - 1.0) if an > 1 else math.inf
        return _HypothesisState(
            model=model,
            changed=changed,
            start_index=start_index,
            start_t=start_t,
            log_evidence=float(log_evidence),
            log_prior=log_prior,
            method="conjugate",
            center=center,
            scale=scale,
            parameter_names=names,
            posterior_mean=bn,
            posterior_noise_variance=float(noise_mean * scale**2),
            g_value=g_value,
            design_condition_number=condition_number,
            vn=vn,
            an=float(an),
            dn=float(dn),
        )

    def _design_matrix(
        self,
        n: int,
        *,
        model: str,
        changed: bool,
        start_index: int | None,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        if not changed:
            return self._basis(model, np.arange(n))
        assert start_index is not None
        pre_age = np.arange(start_index, dtype=float)
        post_age = np.arange(n - start_index, dtype=float)
        pre = np.column_stack([np.ones(start_index), pre_age])
        post, current_names = self._basis(model, post_age)
        design = np.zeros((n, 2 + post.shape[1]))
        design[:start_index, :2] = pre
        design[start_index:, 2:] = post
        names = ("pre_intercept", "pre_slope") + current_names
        return design, names

    def _basis(
        self,
        model: str,
        age: np.ndarray,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        age = np.asarray(age, dtype=float)
        if model == "constant":
            return np.ones((len(age), 1)), ("level",)
        if model == "linear":
            return np.column_stack([np.ones(len(age)), age]), (
                "intercept",
                "slope",
            )
        if model == "logarithmic":
            return np.column_stack([np.ones(len(age)), np.log1p(age)]), (
                "intercept",
                "log_slope",
            )
        raise ValueError(f"base nao linear solicitada: {model}")

    def _fit_exponential_laplace(
        self,
        z: np.ndarray,
        center: float,
        scale: float,
        *,
        changed: bool,
        start_index: int | None,
        start_t: Any | None,
        log_prior: float,
    ) -> _HypothesisState:
        n = len(z)
        if changed:
            assert start_index is not None
            pre_age = np.arange(start_index, dtype=float)
            post_age = np.arange(n - start_index, dtype=float)
            pre_params = np.linalg.lstsq(
                np.column_stack([np.ones(start_index), pre_age]),
                z[:start_index],
                rcond=None,
            )[0]
            log_post = np.log(np.maximum(z[start_index:], 1e-12))
            exp_params = np.linalg.lstsq(
                np.column_stack([np.ones(len(post_age)), post_age]),
                log_post,
                rcond=None,
            )[0]
            exp_params[1] = np.clip(
                exp_params[1],
                -self.config.max_abs_exponential_rate,
                self.config.max_abs_exponential_rate,
            )
            coeff0 = np.r_[pre_params, exp_params]
            names = ("pre_intercept", "pre_slope", "log_amplitude", "log_rate")
            variances = np.array(
                [
                    self.config.prior_intercept_variance,
                    self.config.prior_slope_variance,
                    self.config.prior_log_amplitude_variance,
                    self.config.prior_exponential_rate_variance,
                ]
            )
        else:
            age = np.arange(n, dtype=float)
            log_z = np.log(np.maximum(z, 1e-12))
            coeff0 = np.linalg.lstsq(
                np.column_stack([np.ones(n), age]), log_z, rcond=None
            )[0]
            coeff0[1] = np.clip(
                coeff0[1],
                -self.config.max_abs_exponential_rate,
                self.config.max_abs_exponential_rate,
            )
            names = ("log_amplitude", "log_rate")
            variances = np.array(
                [
                    self.config.prior_log_amplitude_variance,
                    self.config.prior_exponential_rate_variance,
                ]
            )

        def mean(coefficients: np.ndarray) -> np.ndarray:
            if changed:
                assert start_index is not None
                result = np.empty(n)
                result[:start_index] = coefficients[0] + coefficients[1] * np.arange(
                    start_index
                )
                result[start_index:] = np.exp(
                    coefficients[2]
                    + coefficients[3] * np.arange(n - start_index)
                )
                return result
            return np.exp(coefficients[0] + coefficients[1] * np.arange(n))

        initial_residuals = z - mean(coeff0)
        initial_variance = max(float(initial_residuals @ initial_residuals / n), 1e-4)
        initial = np.r_[coeff0, math.log(initial_variance)]
        inv_variances = 1.0 / variances
        logdet_v0 = float(np.sum(np.log(variances)))
        p = len(coeff0)

        def log_joint_v(parameter: np.ndarray) -> float:
            coefficients = parameter[:-1]
            log_variance = float(parameter[-1])
            variance = math.exp(log_variance)
            residuals = z - mean(coefficients)
            rss = float(residuals @ residuals)
            coefficient_quadratic = float(coefficients @ (inv_variances * coefficients))
            log_likelihood = -0.5 * n * (
                math.log(2.0 * math.pi) + log_variance
            ) - rss / (2.0 * variance)
            # Para os parametros nao lineares, a prior Normal e independente
            # da variancia residual. Isso evita que ruido muito baixo force
            # artificialmente log-amplitude e taxa para zero.
            log_coefficient_prior = -0.5 * (
                p * math.log(2.0 * math.pi)
                + logdet_v0
                + coefficient_quadratic
            )
            # Densidade da Inverse-Gamma transformada de sigma^2 para log(sigma^2).
            log_variance_prior = (
                self.config.noise_shape * math.log(self.config.noise_scale)
                - gammaln(self.config.noise_shape)
                - self.config.noise_shape * log_variance
                - self.config.noise_scale / variance
            )
            return float(log_likelihood + log_coefficient_prior + log_variance_prior)

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * len(initial)
        rate_index = 3 if changed else 1
        bounds[rate_index] = (
            -self.config.max_abs_exponential_rate,
            self.config.max_abs_exponential_rate,
        )
        bounds[-1] = (-20.0, 10.0)
        result = minimize(
            lambda parameter: -log_joint_v(parameter),
            initial,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"Laplace exponencial falhou: {result.message}")
        hessian = _finite_difference_hessian(
            lambda parameter: -log_joint_v(parameter), result.x
        )
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        clipped = np.maximum(eigenvalues, 1e-8)
        hessian_pd = eigenvectors @ np.diag(clipped) @ eigenvectors.T
        covariance = np.linalg.inv(hessian_pd)
        logdet_hessian = float(np.sum(np.log(clipped)))
        dimension = len(result.x)
        log_evidence = (
            log_joint_v(result.x)
            + 0.5 * dimension * math.log(2.0 * math.pi)
            - 0.5 * logdet_hessian
        )
        noise_variance = scale**2 * math.exp(float(result.x[-1]))
        return _HypothesisState(
            model="exponential",
            changed=changed,
            start_index=start_index,
            start_t=start_t,
            log_evidence=float(log_evidence),
            log_prior=log_prior,
            method="laplace",
            center=center,
            scale=scale,
            parameter_names=names,
            posterior_mean=result.x,
            posterior_noise_variance=float(noise_variance),
            g_value=None,
            design_condition_number=None,
            laplace_covariance=covariance,
        )

    def _normalize_hypotheses(self, states: list[_HypothesisState]) -> None:
        log_weights = np.asarray(
            [state.log_evidence + state.log_prior for state in states], dtype=float
        )
        normalization = float(logsumexp(log_weights))
        for state, log_weight in zip(states, log_weights):
            state.probability = float(math.exp(log_weight - normalization))

    def _g_sensitivity(
        self,
        observations: tuple[Observation, ...],
        z: np.ndarray,
        center: float,
        scale: float,
        model_priors: dict[str, float],
        *,
        base_states: list[_HypothesisState],
    ) -> tuple[GSensitivityResult, ...]:
        """Repete a inferencia em uma pequena vizinhanca da forca da prior."""

        multipliers = list(dict.fromkeys(
            (*self.config.g_sensitivity_multipliers, self.config.g_prior_multiplier)
        ))
        output: list[GSensitivityResult] = []
        for multiplier in multipliers:
            if math.isclose(multiplier, self.config.g_prior_multiplier):
                states = base_states
            else:
                states = self._enumerate_hypotheses(
                    observations,
                    z,
                    center,
                    scale,
                    model_priors,
                    g_multiplier=multiplier,
                )
                self._normalize_hypotheses(states)
            windows = self._window_probabilities(states, observations)
            models = self._model_probabilities(states)
            best_window = max(windows, key=lambda item: item.probability)
            best_model = max(models, key=models.get)
            weights = np.asarray([state.probability for state in states], dtype=float)
            forecast_means: list[float] = []
            for horizon in range(1, self.config.forecast_horizon + 1):
                component_means = [
                    self._predictive_moments(
                        state,
                        len(observations),
                        horizon,
                        np.random.default_rng(
                            self.config.random_seed + 30_000 + horizon
                        ),
                    )[0]
                    for state in states
                ]
                forecast_means.append(float(weights @ np.asarray(component_means)))
            output.append(
                GSensitivityResult(
                    g_multiplier=float(multiplier),
                    g_value=float(len(z) * multiplier),
                    recommended_window_start_t=best_window.start_t,
                    recommended_no_change=not best_window.changed,
                    recommended_model=best_model,
                    probability_no_change=float(
                        sum(state.probability for state in states if not state.changed)
                    ),
                    model_probabilities=models,
                    forecast_means=tuple(forecast_means),
                )
            )
        return tuple(output)

    def _public_hypothesis(self, state: _HypothesisState) -> HypothesisResult:
        values = state.posterior_mean[:-1] if state.method == "laplace" else state.posterior_mean
        parameters: dict[str, float] = {}
        for name, value in zip(state.parameter_names, values):
            if name == "log_amplitude":
                parameters["amplitude"] = float(state.scale * math.exp(value))
            elif name == "log_rate":
                parameters["log_rate"] = float(value)
            elif name in {"level", "intercept", "pre_intercept"}:
                parameters[name] = float(state.center + state.scale * value)
            else:
                parameters[name] = float(state.scale * value)
        return HypothesisResult(
            model=state.model,
            changed=state.changed,
            start_index=state.start_index,
            start_t=state.start_t,
            log_evidence=state.log_evidence,
            log_prior=state.log_prior,
            posterior_probability=state.probability,
            inference_method=state.method,
            posterior_parameters=parameters,
            posterior_noise_variance=state.posterior_noise_variance,
            g_value=state.g_value,
            design_condition_number=state.design_condition_number,
        )

    def _window_probabilities(
        self,
        states: list[_HypothesisState],
        observations: tuple[Observation, ...],
    ) -> tuple[WindowProbability, ...]:
        no_change = sum(state.probability for state in states if not state.changed)
        values = [
            WindowProbability(False, None, None, float(no_change))
        ]
        starts = sorted(
            {state.start_index for state in states if state.changed and state.start_index is not None}
        )
        for start in starts:
            probability = sum(
                state.probability for state in states if state.start_index == start
            )
            values.append(
                WindowProbability(True, start, observations[start].t, float(probability))
            )
        return tuple(values)

    def _model_probabilities(self, states: list[_HypothesisState]) -> dict[str, float]:
        models = dict.fromkeys((state.model for state in states), 0.0)
        for state in states:
            models[state.model] += state.probability
        return {model: float(probability) for model, probability in models.items()}

    def _credible_set(
        self,
        windows: tuple[WindowProbability, ...],
    ) -> tuple[Any | None, ...]:
        ordered = sorted(windows, key=lambda item: item.probability, reverse=True)
        chosen: list[Any | None] = []
        cumulative = 0.0
        for item in ordered:
            chosen.append(item.start_t)
            cumulative += item.probability
            if cumulative >= self.config.credible_mass:
                break
        return tuple(chosen)

    def _selected_window_states(
        self,
        states: list[_HypothesisState],
        observations: tuple[Observation, ...],
        best_window: WindowProbability,
        *,
        window_start: Any | None,
        force_no_change: bool,
    ) -> tuple[list[_HypothesisState], Any | None, bool, str]:
        if force_no_change:
            return (
                [state for state in states if not state.changed],
                None,
                True,
                "manual",
            )
        if window_start is not None:
            matches = [
                index
                for index, item in enumerate(observations)
                if item.t == window_start
            ]
            if len(matches) != 1:
                raise ValueError("window_start deve identificar exatamente um ponto")
            start = matches[0]
            matching_states = [state for state in states if state.start_index == start]
            if not matching_states:
                raise ValueError("window_start nao e elegivel pelas restricoes minimas")
            return matching_states, window_start, False, "manual"
        if best_window.changed:
            return (
                [state for state in states if state.start_index == best_window.start_index],
                best_window.start_t,
                False,
                "automatic",
            )
        return (
            [state for state in states if not state.changed],
            None,
            True,
            "automatic",
        )

    def _mixture_forecast(
        self,
        states: list[_HypothesisState],
        observations: tuple[Observation, ...],
        rng: np.random.Generator,
        *,
        renormalize: bool = False,
    ) -> tuple[PredictivePoint, ...]:
        weights = np.asarray([state.probability for state in states], dtype=float)
        if renormalize or not np.isclose(weights.sum(), 1.0):
            weights = weights / weights.sum()
        counts = rng.multinomial(self.config.posterior_draws, weights)
        alpha = 1.0 - self.config.confidence
        labels = [item.t for item in observations]
        output: list[PredictivePoint] = []

        for horizon in range(1, self.config.forecast_horizon + 1):
            component_means: list[float] = []
            component_variances: list[float] = []
            mixture_parts: list[np.ndarray] = []
            for state, weight, count in zip(states, weights, counts):
                mean, variance = self._predictive_moments(state, len(observations), horizon, rng)
                component_means.append(mean)
                component_variances.append(variance)
                if count:
                    mixture_parts.append(
                        self._predictive_samples(state, len(observations), horizon, count, rng)
                    )
            samples = np.concatenate(mixture_parts)
            means = np.asarray(component_means)
            variances = np.asarray(component_variances)
            mixture_mean = float(weights @ means)
            within = float(weights @ variances)
            between = float(weights @ ((means - mixture_mean) ** 2))
            output.append(
                PredictivePoint(
                    horizon=horizon,
                    t=_future_label(labels, horizon),
                    mean=mixture_mean,
                    median=float(np.median(samples)),
                    lower=float(np.quantile(samples, alpha / 2.0)),
                    upper=float(np.quantile(samples, 1.0 - alpha / 2.0)),
                    within_hypothesis_variance=within,
                    between_hypothesis_variance=between,
                    total_variance=within + between,
                )
            )
        return tuple(output)

    def _predictive_moments(
        self,
        state: _HypothesisState,
        n: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> tuple[float, float]:
        if state.method == "conjugate":
            row = self._future_row(state, n, horizon)
            assert state.vn is not None and state.an is not None and state.dn is not None
            location = float(row @ state.posterior_mean)
            scale_squared = float(
                state.dn / state.an * (1.0 + row @ state.vn @ row)
            )
            degrees = 2.0 * state.an
            variance = scale_squared * degrees / (degrees - 2.0)
            return state.center + state.scale * location, state.scale**2 * variance

        samples = self._predictive_samples(
            state,
            n,
            horizon,
            self.config.laplace_summary_draws,
            rng,
        )
        return float(np.mean(samples)), float(np.var(samples, ddof=1))

    def _predictive_samples(
        self,
        state: _HypothesisState,
        n: int,
        horizon: int,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if state.method == "conjugate":
            row = self._future_row(state, n, horizon)
            assert state.vn is not None and state.an is not None and state.dn is not None
            location = float(row @ state.posterior_mean)
            scale_squared = float(
                state.dn / state.an * (1.0 + row @ state.vn @ row)
            )
            draws = location + math.sqrt(scale_squared) * rng.standard_t(
                2.0 * state.an, size=count
            )
            return state.center + state.scale * draws

        assert state.laplace_covariance is not None
        parameters = rng.multivariate_normal(
            state.posterior_mean,
            state.laplace_covariance,
            size=count,
            check_valid="ignore",
        )
        rate_index = 3 if state.changed else 1
        parameters[:, rate_index] = np.clip(
            parameters[:, rate_index],
            -self.config.max_abs_exponential_rate,
            self.config.max_abs_exponential_rate,
        )
        parameters[:, -1] = np.clip(parameters[:, -1], -20.0, 10.0)
        if state.changed:
            assert state.start_index is not None
            age = n - state.start_index - 1 + horizon
            means = np.exp(parameters[:, 2] + parameters[:, 3] * age)
        else:
            age = n - 1 + horizon
            means = np.exp(parameters[:, 0] + parameters[:, 1] * age)
        noise = rng.normal(0.0, np.sqrt(np.exp(parameters[:, -1])))
        return state.center + state.scale * (means + noise)

    def _future_row(self, state: _HypothesisState, n: int, horizon: int) -> np.ndarray:
        if state.changed:
            assert state.start_index is not None
            age = n - state.start_index - 1 + horizon
            basis, _ = self._basis(state.model, np.asarray([age]))
            return np.r_[np.zeros(2), basis[0]]
        age = n - 1 + horizon
        basis, _ = self._basis(state.model, np.asarray([age]))
        return basis[0]

    def _current_fitted(
        self,
        state: _HypothesisState,
        observations: tuple[Observation, ...],
    ) -> tuple[FittedPoint, ...]:
        start = state.start_index if state.changed and state.start_index is not None else 0
        count = len(observations) - start
        age = np.arange(count, dtype=float)
        if state.method == "conjugate":
            basis, _ = self._basis(state.model, age)
            coefficients = state.posterior_mean[2:] if state.changed else state.posterior_mean
            values = state.center + state.scale * (basis @ coefficients)
        else:
            coefficients = state.posterior_mean
            if state.changed:
                values = state.center + state.scale * np.exp(
                    coefficients[2] + coefficients[3] * age
                )
            else:
                values = state.center + state.scale * np.exp(
                    coefficients[0] + coefficients[1] * age
                )
        return tuple(
            FittedPoint(t=observation.t, value=float(value))
            for observation, value in zip(observations[start:], values)
        )


def _finite_difference_hessian(function: Any, point: np.ndarray) -> np.ndarray:
    point = np.asarray(point, dtype=float)
    size = len(point)
    hessian = np.zeros((size, size), dtype=float)
    steps = 1e-4 * (1.0 + np.abs(point))
    center = float(function(point))
    for i in range(size):
        ei = np.zeros(size)
        ei[i] = steps[i]
        hessian[i, i] = (
            function(point + ei) - 2.0 * center + function(point - ei)
        ) / (steps[i] ** 2)
        for j in range(i + 1, size):
            ej = np.zeros(size)
            ej[j] = steps[j]
            value = (
                function(point + ei + ej)
                - function(point + ei - ej)
                - function(point - ei + ej)
                + function(point - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _parse_observation(item: Observation | Mapping[str, Any] | Sequence[Any]) -> Observation:
    if isinstance(item, Observation):
        return item
    if isinstance(item, Mapping):
        value = item.get("value", item.get("valor"))
        if value is None or "t" not in item:
            raise ValueError("dict deve conter 'value' ou 'valor', alem de 't'")
        return Observation(float(value), item["t"])
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
        value, label = item
        return Observation(float(value), label)
    raise TypeError("observacao deve ser Observation, dict ou tupla (valor, t)")


def _future_label(labels: list[Any], horizon: int) -> Any | None:
    if len(labels) < 2:
        return None
    previous, last = labels[-2], labels[-1]
    if isinstance(last, Real) and isinstance(previous, Real):
        return last + (last - previous) * horizon
    if isinstance(last, datetime):
        return _add_months(last, horizon)
    if isinstance(last, date):
        return _add_months(last, horizon)
    if isinstance(last, str):
        try:
            parsed = datetime.strptime(last, "%Y-%m").date()
        except ValueError:
            return None
        return _add_months(parsed, horizon).strftime("%Y-%m")
    return None


def _add_months(value: date | datetime, months: int) -> date | datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
