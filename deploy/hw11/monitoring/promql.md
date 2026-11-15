# PromQL стенда ДЗ-11

Клиент измеряет TTFC и полную длительность монотонными часами. Backend-метрики
относятся только к успешно завершённым ответам. Request/session ID остаются
в JSON-журналах; метки метрик ограничены именем Pod, outcome и hit/miss.

Перцентиль classic histogram вычисляется после суммирования bucket-счётчиков
с сохранением `le`. Разрез `pod` сохраняется для диагностики горячей реплики.

```promql
histogram_quantile(0.95,
  sum by (le, pod) (rate(hw11_ttfc_seconds_bucket[1m])))
histogram_quantile(0.95,
  sum by (le, pod) (rate(hw11_request_duration_seconds_bucket[1m])))
sum by (pod) (rate(hw11_requests_total{outcome="completed"}[1m]))
sum by (pod, outcome) (rate(hw11_requests_total{outcome!="completed"}[1m]))
hw11_running
hw11_waiting
sum by (pod) (rate(container_cpu_usage_seconds_total{
  namespace="hw11", container="api", pod=~"hw11-api-.+"}[1m]))
sum by (pod) (container_memory_working_set_bytes{
  namespace="hw11", container="api", pod=~"hw11-api-.+"})
sum(kube_pod_status_ready{
  namespace="hw11", pod=~"hw11-api-.+", condition="true"})
```

Окно `rate` даёт скользящую оценку. Для прямого сравнения одной выборки
с клиентом используются два сохранённых снимка cumulative buckets до arrivals
и после завершения ответов, без перезапусков Pod и чужого трафика между ними.
Разность `_count` обязана совпасть с числом успешных ответов клиента.
Разность buckets агрегируется по `le`; интерполяция внутри bucket даёт
приближённый p50/p95. Клиентские перцентили вычисляются по сырым длительностям.
Они включают сеть и клиентский overhead, поэтому не обязаны совпасть точно.

Два правила находятся в [alerts.yml](alerts.yml): p95 TTFC >0,8с при минимум
10 успешных завершениях за 30 с; доля отказов >5% при минимум 20 терминальных
исходах за 30 с. `for: 10s`, evaluation interval 5 с. Переходы pending/firing/normal
проверяются по `/api/v1/rules`, а не имитируются записями в отчёте.
