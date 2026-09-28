from probabilistic_window_forecast import BayesianConfig, BayesianForecastPipeline


noise = [0.3, -0.4, 0.2, -0.1, 0.5, -0.2]
series = []

for index in range(42):
    if index < 22:
        value = 100 + 0.20 * index + noise[index % len(noise)]
    else:
        value = 118 + 2.40 * (index - 22) + noise[index % len(noise)]
    year = 2022 + index // 12
    month = index % 12 + 1
    series.append((value, f"{year:04d}-{month:02d}"))


pipeline = BayesianForecastPipeline(
    BayesianConfig(
        min_before=8,
        min_after=8,
        forecast_horizon=6,
        posterior_draws=12_000,
    )
)

result = pipeline.run(series, series_id="serie_exemplo")

print("P(sem ruptura):", round(result.probability_no_change, 4))
print("P(com ruptura):", round(result.probability_change, 4))
print("Janela recomendada:", result.recommended_window_start_t)
print("Modelo recomendado:", result.recommended_model)
print("Probabilidades dos modelos:", result.model_probabilities)
print("Centro e escala robustos:", result.center, result.scale)
print("g usado:", result.g_value)
print("Sensivel a g:", result.prior_sensitive)
for sensitivity in result.g_sensitivity_results:
    print(
        "  g multiplicador=",
        sensitivity.g_multiplier,
        "janela=",
        sensitivity.recommended_window_start_t,
        "modelo=",
        sensitivity.recommended_model,
    )

for point in result.forecast:
    print(
        point.t,
        round(point.mean, 2),
        f"[{point.lower:.2f}, {point.upper:.2f}]",
        "var. entre =",
        round(point.between_hypothesis_variance, 3),
    )

# A inferencia automatica e preservada, mas a escolha oficial pode ser humana.
reviewed = pipeline.run(
    series,
    series_id="serie_exemplo",
    window_start="2024-01",
    model_override="linear",
)

print("Janela escolhida:", reviewed.selected_window_start_t)
print("Modelo escolhido:", reviewed.selected_model)
print("Intervencao resolvida:", reviewed.intervention_resolved_by_user)
