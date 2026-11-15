"""Rebuild the HW11 report from preserved measurements; never invent missing runs."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
import os
import tempfile
from pathlib import Path
import re
import statistics

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mlops-hw11-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_ERRORS = []
OUTCOMES = ("completed", "rejected", "interrupted", "timeout", "connection_error", "error", "unsent")
COLORS = ("#4c5cbe", "#d47724", "#36836b", "#a74d90")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.grid": True, "grid.alpha": .2})


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default
    except json.JSONDecodeError as exc:
        DATA_ERRORS.append(f"{path}: invalid JSON at line {exc.lineno}")
        return {} if default is None else default


def rows(path):
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return []
    lines = text.splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            # Only a partial final line of a still-running collector is tolerated.
            if index != len(lines) - 1 or text.endswith("\n"):
                DATA_ERRORS.append(f"{path}: invalid JSONL at line {index + 1}")
    return result


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def percentile(values, probability):
    values = sorted(value for value in values if finite(value))
    if not values:
        return None
    rank = (len(values) - 1) * probability
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (rank - low)


def distribution(values):
    values = [value for value in values if finite(value)]
    return dict(n=len(values), p50=percentile(values, .5), p95=percentile(values, .95),
                maximum=max(values) if values else None)


def summarize(data):
    counts = Counter(row.get("outcome", "unknown") for row in data)
    completed = [row for row in data if row.get("outcome") == "completed"]
    sent = len(data) - counts["unsent"]
    errors = sum(counts[key] for key in ("interrupted", "connection_error", "error"))
    hits = sum(row.get("cache_hit") is True for row in completed)
    misses = sum(row.get("cache_hit") is False for row in completed)
    pods = {}
    for pod in sorted({row.get("pod") for row in data if row.get("pod")}):
        selected = [row for row in data if row.get("pod") == pod]
        good = [row for row in selected if row.get("outcome") == "completed"]
        pods[pod] = dict(attempts=len(selected), successful=len(good),
                         ttfc_seconds=distribution([r.get("ttfc_seconds") for r in good]),
                         queue_seconds=distribution([r.get("queue_seconds") for r in good]))
    ids = [row.get("attempt_id") for row in data]
    violations = []
    if len(set(ids)) != len(ids):
        violations.append("duplicate or missing attempt IDs")
    for row in completed:
        if not (row.get("status") == 200 and row.get("final_marker") and row.get("chunks", 0) > 0):
            violations.append("invalid successful response")
        if row.get("error") is not None:
            violations.append("successful response has a transport/protocol error")
        if row.get("chunks") != row.get("payload", {}).get("max_chunks") or row.get("chunks") != row.get("done", {}).get("chunks"):
            violations.append("successful response has an unexpected chunk count")
        if len(row.get("inter_chunk_seconds", [])) != row.get("chunks", 0) - 1:
            violations.append("wrong inter-chunk denominator")
        if row.get("chunks") == 1 and row.get("inter_chunk_latency_seconds") is not None:
            violations.append("single-chunk response has inter-chunk latency")
        if row.get("backend_mode") != "simulated":
            violations.append("unexpected backend mode")
        if not all(row.get(key) for key in ("request_id", "pod", "model_version", "artifact_sha256")):
            # Client ID is also preserved in payload/attempt_id by older raw clients.
            if not all(row.get(key) for key in ("pod", "model_version", "artifact_sha256")) or not row.get("payload", {}).get("request_id"):
                violations.append("missing actual response identity")
    return dict(planned=len(data), sent=sent, successful=len(completed),
                cache=dict(hits=hits, misses=misses, unknown=len(completed)-hits-misses,
                           hit_fraction_of_known=hits/(hits+misses) if hits+misses else None), counts={key: counts[key] for key in OUTCOMES},
                error_fraction_of_sent=errors / sent if sent else None,
                timeout_fraction_of_sent=counts["timeout"] / sent if sent else None,
                rejection_fraction_of_sent=counts["rejected"] / sent if sent else None,
                unsuccessful_fraction_of_sent=(sent - len(completed)) / sent if sent else None,
                unsent_fraction_of_planned=counts["unsent"] / len(data) if data else None,
                ttfc_seconds=distribution([row.get("ttfc_seconds") for row in completed]),
                full_seconds=distribution([row.get("full_seconds") for row in completed]),
                send_lag_seconds=distribution([row.get("send_lag_seconds") for row in data]),
                inter_chunk_seconds=distribution([x for row in completed for x in row.get("inter_chunk_seconds", [])]),
                server_ttfc_seconds=distribution([row.get("done", {}).get("ttfc_seconds") for row in completed]),
                server_full_seconds=distribution([row.get("done", {}).get("total_seconds") for row in completed]),
                per_pod=pods, violations=sorted(set(violations)))


def prometheus_result(document):
    while isinstance(document, dict) and "result" not in document:
        if not isinstance(document.get("data"), dict):
            return []
        document = document["data"]
    return document.get("result", []) if isinstance(document, dict) else []


def metric_series(path):
    result = []
    for series in prometheus_result(read(path)):
        values = [(float(t), float(value)) for t, value in series.get("values", []) if finite(float(value))]
        if values:
            result.append((series.get("metric", {}).get("pod", "all"), values))
    return result


def timeline(path):
    points = []
    for sample in rows(path):
        ready = endpoints = hpa = 0
        for item in sample.get("data", {}).get("items", []):
            kind = item.get("kind")
            if kind == "Pod" and item.get("metadata", {}).get("labels", {}).get("app") == "hw11-api":
                ready += any(c["type"] == "Ready" and c["status"] == "True" for c in item.get("status", {}).get("conditions", []))
            elif kind == "EndpointSlice" and item.get("metadata", {}).get("labels", {}).get("kubernetes.io/service-name") == "hw11-api":
                endpoints += sum(ep.get("conditions", {}).get("ready") is True for ep in item.get("endpoints", []))
            elif kind == "HorizontalPodAutoscaler":
                hpa = item.get("status", {}).get("currentReplicas", 0)
        points.append((sample["utc"], ready, endpoints, hpa))
    return points


def parse_histogram(text, name):
    buckets = {}
    count = None
    for line in text.splitlines():
        if line.startswith(name + "_bucket{"):
            match = re.search(r'le="([^"]+)"', line)
            if match:
                buckets[float(match.group(1))] = float(line.split()[-1])
        elif line.startswith(name + "_count "):
            count = float(line.split()[-1])
    return buckets, count


def histogram_quantile(buckets, probability):
    if not buckets or buckets.get(math.inf, 0) <= 0:
        return None
    rank = buckets[math.inf] * probability
    lower, previous = 0., 0.
    for upper, count in sorted(buckets.items()):
        if count >= rank:
            if upper == math.inf:
                return lower
            return lower + (upper - lower) * (rank - previous) / (count - previous) if count > previous else upper
        lower, previous = upper, count
    return None


def cohort(path, summary):
    before, after = read(path / "metrics-before.json"), read(path / "metrics-after.json")
    if not before or not after:
        return dict(available=False)
    same = set(before.get("pods", {})) == set(after.get("pods", {}))
    result = dict(available=True, same_pods=same, before_utc=before.get("utc"), after_utc=after.get("utc"))
    for metric, key in (("hw11_ttfc_seconds", "ttfc"), ("hw11_request_duration_seconds", "full")):
        delta, count, valid = defaultdict(float), 0., same
        for pod in before.get("pods", {}):
            old, old_n = parse_histogram(before["pods"][pod], metric)
            new, new_n = parse_histogram(after.get("pods", {}).get(pod, ""), metric)
            if old_n is None or new_n is None or new_n < old_n or old.keys() != new.keys():
                valid = False
                continue
            count += new_n - old_n
            for upper in old:
                difference = new[upper] - old[upper]
                valid = valid and difference >= 0
                delta[upper] += difference
        valid = valid and count == summary["successful"] and delta.get(math.inf) == count
        result[key] = dict(valid=valid, delta_count=count, client_successful=summary["successful"],
                           buckets={str(upper): value for upper, value in sorted(delta.items())},
                           histogram_p50_seconds=histogram_quantile(delta, .5) if valid else None,
                           histogram_p95_seconds=histogram_quantile(delta, .95) if valid else None,
                           client=summary[key + "_seconds"], server_exact=summary["server_" + key + "_seconds"])
    prom_before, prom_after = read(path / "prom-before.json"), read(path / "prom-after.json")
    prom = dict(available=bool(prom_before and prom_after),
                matched_direct_counts=prom_before.get("matched_direct_counts") is True and prom_after.get("matched_direct_counts") is True)
    for metric, key in (("hw11_ttfc_seconds", "ttfc"), ("hw11_request_duration_seconds", "full")):
        snapshots = []
        for document in (prom_before, prom_after):
            collected = {}
            for entry in prometheus_result(document):
                labels = entry.get("metric", {})
                pod = labels.get("pod")
                name = labels.get("__name__")
                if pod not in before.get("pods", {}) or name not in (metric + "_bucket", metric + "_count"):
                    continue
                collected[(pod, name, labels.get("le"))] = float(entry["value"][1])
            snapshots.append(collected)
        old, new = snapshots
        valid = prom["available"] and prom["matched_direct_counts"] and bool(old) and old.keys() == new.keys()
        delta, count = defaultdict(float), 0.
        for labels, value in old.items():
            difference = new.get(labels, -math.inf) - value
            if not finite(difference) or difference < 0:
                valid = False
                continue
            if labels[1].endswith("_count"):
                count += difference
            else:
                delta[float(labels[2])] += difference
        direct = result[key]
        expected = {float(upper): value for upper, value in direct["buckets"].items()}
        valid = valid and direct["valid"] and count == direct["delta_count"] and dict(delta) == expected
        prom[key] = dict(valid=valid, delta_count=count,
                         histogram_p50_seconds=histogram_quantile(delta, .5) if valid else None,
                         histogram_p95_seconds=histogram_quantile(delta, .95) if valid else None)
    result["prometheus"] = prom
    result["prometheus_snapshot_files"] = [p.name for p in sorted(path.glob("prom-before.json")) + sorted(path.glob("prom-after.json"))]

    return result


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + ".png"), dpi=145, bbox_inches="tight")
    svg = directory / (name + ".svg")
    fig.savefig(svg, bbox_inches="tight")
    # Matplotlib inserts trailing spaces in multiline SVG path attributes.
    # Newlines keep the same token separation after trimming these spaces.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def empty(axis, text):
    axis.text(.5, .5, text, ha="center", va="center", transform=axis.transAxes)


def figure_a(base, output):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    proof = read(base / "a/drain-proof/proof.json")
    rollout_name = "drain-proof" if proof.get("passed") is True else "rollout"
    rollout_title = (f"Активные потоки при удалении Pod · warmup {proof['warmup_seconds']} с"
                     if rollout_name == "drain-proof" else "Обновление A → B · warmup 20 с")
    for axis, name, title in zip(axes, ("faulty", "fixed", rollout_name),
                                 ("Ошибочные probes · warmup 20 с", "Исправленные probes · warmup 20 с", rollout_title)):
        path = base / "a" / name
        points = timeline(path / "timeline.jsonl")
        requests = rows(path / "requests.jsonl")
        if name in ("rollout", "drain-proof"):
            requests += rows(path / "background/requests.jsonl")
        if not points:
            empty(axis, "Измерения ещё не сохранены")
        else:
            origin = points[0][0]
            axis.step([p[0] - origin for p in points], [p[1] for p in points], where="post", label="Ready Pod")
            axis.step([p[0] - origin for p in points], [p[2] for p in points], where="post", label="Готовые EndpointSlice адреса", linestyle="--")
            failures = [r for r in requests if r.get("outcome") != "completed" and r.get("sent_utc")]
            if failures:
                axis.scatter([r["sent_utc"] - origin for r in failures], [0] * len(failures), marker="x", color="#bf3940", label=f"Неуспешные попытки (N={len(failures)})")
            if name == "drain-proof":
                for index, stream in enumerate(proof.get("streams", [])):
                    color = COLORS[index % len(COLORS)]
                    if stream.get("first_chunk_utc") and stream.get("end_utc"):
                        axis.axvspan(stream["first_chunk_utc"] - origin, stream["end_utc"] - origin,
                                     alpha=.12, color=color, label=f"Поток {index + 1}: первая часть → EOF")
                    if stream.get("deletion_utc"):
                        axis.axvline(stream["deletion_utc"] - origin, color=color, linestyle="--",
                                     label=f"Начало завершения Pod потока {index + 1}")
                    if stream.get("expected_sigterm_utc"):
                        axis.axvline(stream["expected_sigterm_utc"] - origin, color=color, linestyle=":",
                                     label=f"Начало + preStop {proof['prestop_seconds']} с (поток {index + 1})")
                if proof.get("upgrade_utc"):
                    axis.axvline(proof["upgrade_utc"] - origin, color="black", linestyle="-.", label="Начало обновления")
            long = read(path / "long-stream.json") if name == "rollout" else {}
            if long:
                row = long.get("request", {})
                if row.get("sent_utc") and row.get("end_utc"):
                    axis.axvspan(row["sent_utc"] - origin, row["end_utc"] - origin, alpha=.12, color="green", label="Время длительного потока")
                if long.get("rollout_started_utc"):
                    axis.axvline(long["rollout_started_utc"] - origin, color="black", linestyle=":", label="Начало обновления")
            axis.legend(loc="upper right", fontsize=8)
        axis.set(title=title, xlabel="Секунды от начала наблюдения (UTC)", ylabel="Число готовых реплик")
        axis.set_ylim(bottom=-.2)
    fig.suptitle("A · основные проверки прогрева 20 с; отдельная проверка активного потока при удалении Pod")
    save_figure(fig, output, "01-warmup-rollout")


def figure_b_hot(base, output):
    candidates = sorted((base / "b/clientip/skewed").glob("run-*/requests.jsonl"))
    path = candidates[-1].parent if candidates else base / "b/calibration"
    data = rows(path / "requests.jsonl")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)
    times = [r["sent_utc"] for r in data if r.get("sent_utc")]
    origin = min(times) if times else 0
    pods = sorted({r.get("pod") for r in data if r.get("pod")})
    for index, pod in enumerate(pods):
        selected = [r for r in data if r.get("pod") == pod and r.get("outcome") == "completed"]
        for axis, field in zip(axes[:2], ("ttfc_seconds", "queue_seconds")):
            axis.scatter([r["sent_utc"] - origin for r in selected if finite(r.get(field))],
                         [r[field] for r in selected if finite(r.get(field))], s=10, alpha=.7,
                         color=COLORS[index % len(COLORS)], label=pod)
    for field, linestyle in (("running", "-"), ("waiting", "--")):
        for index, (pod, values) in enumerate(metric_series(path / f"prom-{field}-range.json")):
            axes[2].step([t - origin for t, _ in values], [v for _, v in values], where="post",
                         linestyle=linestyle, color=COLORS[(pods.index(pod) if pod in pods else index) % len(COLORS)], label=f"{pod} {field}")
    for axis in axes:
        if axis.lines or axis.collections:
            axis.legend(fontsize=7, loc="upper left")
        else:
            empty(axis, "Измерения ещё не сохранены")
        axis.set_xlabel("Секунды от первой фактической отправки")
    axes[0].set_ylabel("TTFC успешных ответов, с")
    axes[1].set_ylabel("Очередь успешных ответов, с")
    axes[2].set_ylabel("Запросы running / waiting")
    fig.suptitle(f"B · ClientIP, 80% через одного клиента · {path.relative_to(base)}")
    save_figure(fig, output, "02-hot-replica")


def figure_b_comparison(b_runs, output):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, traffic in enumerate(("uniform", "skewed")):
        for col, (key, title) in enumerate((("ttfc_seconds", "TTFC успешных, p95, с"),
                                             ("full_seconds", "Полная задержка успешных, p95, с"),
                                             ("unsuccessful_fraction_of_sent", "Неуспешные / отправленные, %"))):
            axis = axes[row, col]
            found = False
            for index, policy in enumerate(("normal", "clientip")):
                selected = sorted([run for run in b_runs if run["policy"] == policy and run["traffic"] == traffic], key=lambda run: run["repeat"])
                valid = [(run, run["summary"][key]["p95"] if col < 2 else
                          (run["summary"][key] * 100 if run["summary"][key] is not None else None)) for run in selected]
                valid = [(run, value) for run, value in valid if value is not None]
                if valid:
                    found = True
                    axis.plot([run["repeat"] for run, _ in valid], [value for _, value in valid], "o-", label=policy, color=COLORS[index])
                    for run, value in valid:
                        axis.annotate(f'N={run["summary"]["successful"]}', (run["repeat"], value), xytext=(0, 5 + 8 * index), textcoords="offset points", fontsize=7)
            if not found:
                empty(axis, "Завершённых сравнительных прогонов ещё нет")
            axis.set(title=f"{traffic} · {title}", xlabel="Повтор", xticks=[1, 2, 3])
            axis.set_ylim(bottom=0)
            if found:
                # Leave room for N labels above the highest observed point.
                upper = axis.get_ylim()[1] * 1.18
                axis.set_ylim(0, max(1., upper) if col == 2 else upper)
                axis.legend(fontsize=8)
    fig.suptitle("B · одинаковые расписания, пустой кеш, 2 политики × 2 профиля × 3 повтора")
    save_figure(fig, output, "03-policy-comparison")


def figure_c(base, output):
    path = base / "c/hpa"
    points = timeline(path / "timeline.jsonl")
    data = rows(path / "load/requests.jsonl")
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True, sharex=True)
    if points:
        origin = points[0][0]
        axes[0].step([p[0] - origin for p in points], [p[3] for p in points], where="post", label="Текущее число реплик HPA")
        axes[0].step([p[0] - origin for p in points], [p[1] for p in points], where="post", linestyle="--", label="Готовые API Pod")
        for pod, values in metric_series(path / "prom-cpu-range.json"):
            axes[1].plot([t - origin for t, _ in values], [v for _, v in values], label=pod)
        arrivals = Counter(int(r["sent_utc"] - origin) for r in data if r.get("sent_utc") is not None)
        if arrivals:
            low, high = 0, math.ceil(points[-1][0] - origin)
            axes[2].step(list(range(low, high + 1)), [arrivals[t] for t in range(low, high + 1)], where="post", label="Фактических отправок / с")
        window = read(path / "load-window.json")
        if window.get("stop_utc"):
            for axis in axes:
                axis.axvline(window["stop_utc"] - origin, color="black", linestyle=":", label="Конец расписания отправок")
        for axis in axes:
            axis.legend(fontsize=8, loc="upper right")
    else:
        for axis in axes:
            empty(axis, "Измерения HPA ещё не сохранены")
    for axis, label in zip(axes, ("Число реплик", "CPU API, cores", "Клиентских запросов / с")):
        axis.set(xlabel="Секунды от начала наблюдения (UTC)", ylabel=label)
        axis.set_ylim(bottom=0)
    fig.suptitle("C · фактический HPA 2 → 3 → 2, готовность, CPU и поток запросов")
    save_figure(fig, output, "04-hpa")


def number(value):
    return "—" if value is None else f"{value:.4g}" if isinstance(value, (int, float)) else str(value)


def table(headers, values):
    return "<div class='table'><table><thead><tr>" + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(number(x))}</td>" for x in row) + "</tr>" for row in values) + "</tbody></table></div>"


def link(base, path, label=None):
    path = Path(path)
    if not path.exists():
        return html.escape(label or path.name) + " <em>(ещё нет)</em>"
    import os
    target = os.path.relpath(path, base)
    return f'<a href="{html.escape(target, quote=True)}">{html.escape(label or target)}</a>'


def routing_trace(path, data):
    """Join synthetic client attempts to observed source addresses in Pod logs."""
    attempts = {row.get("request_id") or row.get("payload", {}).get("request_id") or row.get("attempt_id"): row
                for row in data}
    observed = defaultdict(set)
    events = Counter()
    log_files = []
    for logfile in sorted(path.glob("*.log")):
        log_files.append(logfile.name)
        for line in logfile.read_text().splitlines():
            offset = line.find("{")
            if offset < 0:
                continue
            try:
                event = json.loads(line[offset:])
            except json.JSONDecodeError:
                continue
            request_id = event.get("request_id")
            if (request_id in attempts and event.get("event") in ("request_received", "request_finished")
                    and event.get("source_ip")):
                observed[request_id].add(event["source_ip"])
                events[event["event"]] += 1
    grouped = {}
    for request_id, row in attempts.items():
        key = (str(row.get("client")), str(row.get("client_ip")), str(row.get("pod") or "no response Pod"))
        entry = grouped.setdefault(key, dict(client=key[0], client_ip=key[1], pod=key[2], requests=0,
                                             source_ips=set(), matched_request_ids=0, source_mismatch_ids=0))
        entry["requests"] += 1
        entry["source_ips"].update(observed[request_id])
        entry["matched_request_ids"] += bool(observed[request_id])
        entry["source_mismatch_ids"] += bool(observed[request_id] and observed[request_id] != {row.get("client_ip")})
    for entry in grouped.values():
        entry["source_ips"] = sorted(entry["source_ips"])
    return dict(mapping=list(grouped.values()), log_files=log_files, matched_attempts=len([x for x in observed.values() if x]),
                total_attempts=len(data), log_events=dict(events),
                method="request_id joins client payload to request_received/request_finished Pod logs; missing last log samples are not inferred")


def b_catalog(base):
    result = []
    for policy in ("normal", "clientip"):
        for traffic in ("uniform", "skewed"):
            for repeat in range(1, 4):
                path = base / "b" / policy / traffic / f"run-{repeat}"
                if (path / "requests.jsonl").exists():
                    data = rows(path / "requests.jsonl")
                    summary = summarize(data)
                    result.append(dict(policy=policy, traffic=traffic, repeat=repeat, path=str(path.relative_to(base)),
                                       config=read(path / "config.json"), summary=summary,
                                       cohort=cohort(path, summary), routing=routing_trace(path, data)))
    return result


def routing_decision(b_runs):
    result = {}
    for policy in ("normal", "clientip"):
        selected = [run for run in b_runs if run["policy"] == policy and run["traffic"] == "skewed"]
        sent = sum(run["summary"]["sent"] for run in selected)
        result[policy] = dict(repeats=len(selected),
            median_run_p95_ttfc_seconds=percentile([run["summary"]["ttfc_seconds"]["p95"] for run in selected], .5),
            unsuccessful_fraction_of_sent=(sum(run["summary"]["sent"]-run["summary"]["successful"] for run in selected)/sent) if sent else None,
            cache_misses_completed=sum(run["summary"]["cache"]["misses"] for run in selected))
    ready = all(result[policy]["repeats"] == 3 and result[policy]["median_run_p95_ttfc_seconds"] is not None for policy in result)
    choice = None
    if ready:
        normal, sticky = result["normal"], result["clientip"]
        if (normal["median_run_p95_ttfc_seconds"] < sticky["median_run_p95_ttfc_seconds"] and
                normal["unsuccessful_fraction_of_sent"] <= sticky["unsuccessful_fraction_of_sent"]):
            choice = "normal"
    return dict(ready=ready, supported_fix=choice, heavy_profile=result,
                criterion="choose normal only if heavy-profile median run p95 TTFC is lower with no higher unsuccessful/sent ratio; three repeats are descriptive evidence")


def aggregate_requests(base):
    """One canonical ledger; archived attempts and smoke remain separate."""
    sources = []
    for experiment in ("a", "b", "c"):
        for path in sorted((base / experiment).rglob("requests.jsonl")):
            data = rows(path)
            sources.append((path, data))
            schedule_path = path.parent / "schedule.json"
            if schedule_path.exists():
                schedule = read(schedule_path, [])
                expected = Counter(item.get("id") for item in schedule)
                actual = Counter(item.get("id") for item in data)
                if expected != actual:
                    DATA_ERRORS.append(f"{path}: request rows do not match planned schedule")
    for relative in ("a/rollout/long-stream.json", "a/forced-kill/forced-stream.json"):
        path = base / relative
        document = read(path)
        if document:
            if not isinstance(document.get("request"), dict):
                DATA_ERRORS.append(f"{path}: missing request record")
            else:
                sources.append((path, [document["request"]]))
    output = []
    for path, data in sources:
        relative = path.relative_to(base)
        for item in data:
            output.append(dict(item, source_file=str(relative), experiment=relative.parts[0],
                               diagnostic=relative.parts[0] == "b" and relative.parts[1] in ("calibration", "no-skew")))
    ids = [item.get("attempt_id") for item in output]
    errors = []
    if any(not identifier for identifier in ids):
        errors.append("aggregate has a missing attempt ID")
    if len(set(ids)) != len(ids):
        errors.append("aggregate has duplicate attempt IDs")
    if any(item.get("outcome") not in OUTCOMES for item in output):
        errors.append("aggregate contains a request without terminal outcome")
    if len(output) != sum(len(data) for _, data in sources):
        errors.append("aggregate lost source rows")
    DATA_ERRORS.extend(errors)
    output.sort(key=lambda item: (item.get("planned_utc") or 0, item.get("attempt_id") or ""))
    descriptor, temporary = tempfile.mkstemp(prefix=".requests-", suffix=".jsonl.tmp", dir=base)
    try:
        with os.fdopen(descriptor, "w") as stream:
            for item in output:
                stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, base / "requests.jsonl")
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return dict(rows=len(output), source_files=len(sources), unique_attempt_ids=len(set(ids)),
                all_terminal=all(item.get("outcome") in OUTCOMES for item in output),
                per_experiment=dict(Counter(item["experiment"] for item in output)),
                expected_source_rows=sum(len(data) for _, data in sources), errors=errors)


def hpa_summary(base):
    path = base / "c/hpa"
    points = timeline(path / "timeline.jsonl")
    changes = []
    last = None
    for utc, ready, endpoints, replicas in points:
        if replicas != last:
            changes.append(dict(utc=utc, replicas=replicas, ready=ready))
            last = replicas
    pods = {}
    for sample in rows(path / "timeline.jsonl"):
        for pod in sample.get("data", {}).get("items", []):
            if pod.get("kind") != "Pod" or pod.get("metadata", {}).get("labels", {}).get("app") != "hw11-api":
                continue
            metadata = pod["metadata"]
            entry = pods.setdefault(metadata["name"], dict(created_utc=metadata.get("creationTimestamp")))
            for condition in pod.get("status", {}).get("conditions", []):
                if condition["type"] == "Ready" and condition["status"] == "True":
                    entry.setdefault("ready_utc", condition.get("lastTransitionTime"))
    return dict(transitions=changes, pods=pods, scale_down=read(path / "scale-down.json"), load_window=read(path / "load-window.json"))


def hot_hpa_summary(base):
    """Count requests to every observed API Pod, including an unused new replica."""
    path = base / "c/hot-replica"
    data = rows(path / "requests.jsonl")
    samples = rows(path / "timeline.jsonl")
    config = read(path / "config.json")
    if not data or not samples:
        return dict(available=False)
    seen = {}
    initial = None
    for sample in samples:
        current = [pod for pod in sample.get("data", {}).get("items", [])
                   if pod.get("kind") == "Pod" and pod.get("metadata", {}).get("labels", {}).get("app") == "hw11-api"]
        if initial is None and current:
            initial = {pod["metadata"]["name"] for pod in current}
        for pod in current:
            name = pod["metadata"]["name"]
            entry = seen.setdefault(name, dict(pod=name, created_utc=pod["metadata"].get("creationTimestamp"),
                                               ready_utc=None, new_during_observation=name not in (initial or set())))
            for condition in pod.get("status", {}).get("conditions", []):
                if condition["type"] == "Ready" and condition["status"] == "True" and entry["ready_utc"] is None:
                    entry["ready_utc"] = condition.get("lastTransitionTime")
    for name, entry in seen.items():
        selected = [row for row in data if row.get("pod") == name]
        counts = Counter(row.get("outcome") for row in selected)
        entry.update(attempts=len(selected), successful=counts["completed"], rejected=counts["rejected"])
        if entry["created_utc"] and entry["ready_utc"]:
            created = datetime.fromisoformat(entry["created_utc"].replace("Z", "+00:00"))
            ready = datetime.fromisoformat(entry["ready_utc"].replace("Z", "+00:00"))
            entry["creation_to_ready_seconds"] = (ready - created).total_seconds()
    return dict(available=True, service=config.get("service"), duration_seconds=config.get("duration_seconds"),
                sent=sum(row.get("outcome") != "unsent" for row in data),
                attempts_without_response_pod=sum(not row.get("pod") for row in data),
                per_pod=sorted(seen.values(), key=lambda entry: entry["pod"]),
                meaning="attempt counts use observed HTTP response Pod identities; zero is shown for API Pods present in the timeline but receiving no recorded attempts")


def generate(base, render_figures=True):
    DATA_ERRORS.clear()
    base = base.resolve()
    output = base / "figures"
    output.mkdir(parents=True, exist_ok=True)
    b_runs = b_catalog(base)
    checks = read(base / "checks.json")
    aggregate = aggregate_requests(base)
    requests = {}
    # Leaf requests files only; no client-* duplicates and no archived attempts.
    for experiment in ("a", "b", "c"):
        for path in sorted((base / experiment).rglob("requests.jsonl")):
            requests[str(path.relative_to(base))] = summarize(rows(path))
    for relative in ("a/rollout/long-stream.json", "a/forced-kill/forced-stream.json"):
        attempt = read(base / relative).get("request")
        if attempt:
            requests[relative] = summarize([attempt])
    a_checks = checks.get("checks", {}).get("a", {})
    b_checks = checks.get("checks", {}).get("b", {})
    c_checks = checks.get("checks", {}).get("c", {})
    lecture = read(base / "lecture-case.json")
    lecture_complete = all(lecture.get(key) for key in ("lecture_url", "primary_url", "company", "case_text"))
    invariants = [f"{path}: {message}" for path, summary in requests.items() for message in summary["violations"]]
    cohort_errors = [run["path"] for run in b_runs
                     if not run["cohort"].get("available") or
                     not all(run["cohort"].get(key, {}).get("valid") and
                             run["cohort"].get("prometheus", {}).get(key, {}).get("valid")
                             for key in ("ttfc", "full"))]
    regression = read(base / "regression.json")
    missing_required_files = [name for name in ("environment-before.json", "environment-after.json", "image-build.json", "regression.json",
                                               "monitoring-validation.json", "grafana-export.json", "a/drain-proof/proof.json")
                              if not (base / name).exists()]
    drain_proof = read(base / "a/drain-proof/proof.json")
    technical_complete = (checks.get("passed") is True and len(b_runs) == 12 and
                          regression.get("passed") is True and not missing_required_files and
                          read(base / "monitoring-validation.json").get("passed") is True and
                          drain_proof.get("passed") is True and
                          not invariants and not cohort_errors and not DATA_ERRORS)
    complete = technical_complete and lecture_complete
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), complete=complete,
                   routing_decision=routing_decision(b_runs), aggregate=aggregate, technical_complete=technical_complete, backend_mode="simulated", run_summaries=requests, b_runs=b_runs,
                   hpa=hpa_summary(base), hot_hpa=hot_hpa_summary(base), drain_proof=drain_proof,
                   checks=checks, lecture_source_confirmed=lecture_complete,
                   missing_required_files=missing_required_files, data_errors=list(DATA_ERRORS), invariant_violations=invariants, histogram_cohort_errors=cohort_errors,
                   regression=regression)
    if render_figures:
        figure_a(base, output)
        figure_b_hot(base, output)
        figure_b_comparison(b_runs, output)
        figure_c(base, output)
    summary["data_errors"] = sorted(set(DATA_ERRORS))
    summary["technical_complete"] = summary["technical_complete"] and not summary["data_errors"]
    summary["complete"] = summary["technical_complete"] and lecture_complete
    write_html(base, summary, a_checks, b_checks, c_checks, lecture)
    if sorted(set(DATA_ERRORS)) != summary["data_errors"]:
        summary["data_errors"] = sorted(set(DATA_ERRORS))
        summary["technical_complete"] = summary["complete"] = False
        write_html(base, summary, a_checks, b_checks, c_checks, lecture)
    (base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return summary


def write_html(base, summary, a_checks, b_checks, c_checks, lecture):
    def raw(path, title=None):
        return link(base, base / path, title)

    def asset(path, title=None):
        return link(base, ROOT / path, title)

    def status(check):
        return "PASS" if check.get("passed") is True else "НЕ ПОДТВЕРЖДЕНО" if not check else "FAIL"

    def json_block(value):
        return "<pre>" + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

    runs = summary["run_summaries"]
    b_runs = summary["b_runs"]
    before = read(base / "environment-before.json")
    after = read(base / "environment-after.json")
    setup = read(base / "image-build.json") or read(base / "setup-final.json") or read(base / "setup.json")
    drain_proof = summary.get("drain_proof") or read(base / "a/drain-proof/proof.json")
    hot_hpa = summary.get("hot_hpa") or hot_hpa_summary(base)
    node_info = before.get("node", {}).get("items", [{}])[0].get("status", {}).get("nodeInfo", {})
    docker = before.get("docker", {})
    environment = dict(kubernetes=node_info.get("kubeletVersion"), architecture=node_info.get("architecture"),
                       container_runtime=node_info.get("containerRuntimeVersion"), docker=docker.get("ServerVersion"),
                       cpus=docker.get("NCPU"), memory_bytes=docker.get("MemTotal"),
                       host_free_bytes=before.get("host_disk", {}).get("free"), tools=before.get("tools"),
                       actual_vm_disks=before.get("vm", {}).get("stdout"),
                       final_disk=after.get("vm_disk") or after.get("disk"),
                       final_docker_disk=after.get("docker_disk"),
                       image=setup.get("image") or {key: setup.get(key) for key in ("image_index_sha256", "image_config_sha256")},
                       final_pod_image_ids=[dict(pod=pod.get("metadata", {}).get("name"),
                           image_ids=[container.get("imageID") for container in pod.get("status", {}).get("containerStatuses", [])])
                           for pod in after.get("pods", {}).get("items", []) if pod.get("metadata", {}).get("labels", {}).get("app") == "hw11-api"])
    parts = ["<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>ДЗ 11 — Kubernetes под нагрузкой</title><style>"
             "*{box-sizing:border-box}body{margin:0;background:#f3f5fa;color:#202c40;font:17px/1.65 system-ui,sans-serif}"
             "main{max-width:1230px;margin:auto;padding:44px 32px 90px}h1{font-size:40px;line-height:1.15;max-width:850px}"
             "h2{font-size:27px;margin-top:46px}h3{font-size:21px;margin-top:30px}a{color:#364f9c}code,pre{font-family:ui-monospace,monospace;font-size:.85em}"
             "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#e7edf6;padding:18px;border-radius:8px}.lead{font-size:21px}"
             ".notice{padding:18px 22px;border-left:5px solid #bf7c20;background:#fff1d9}.ok{border-color:#35856b;background:#e5f3ed}"
             ".table{overflow-x:auto;background:white;border-radius:8px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}"
             "td,th{padding:10px 12px;border-bottom:1px solid #dce3ed;text-align:left;vertical-align:top}th{background:#e8eef8}"
             "figure{margin:28px 0;background:white;padding:12px;border-radius:10px}figure img{width:100%;height:auto}figcaption{color:#516174;font-size:14px}"
             ".muted{color:#526174}.tag{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:#526174}li{margin:7px 0}"
             "@media print{body{background:white}main{padding:0}h2{break-after:avoid}figure,table{break-inside:avoid}a{color:inherit}}"
             "</style></head><body><main><p class='tag'>MLOPS-26 · ДЗ 11 · CPU-стенд</p>",
             "<h1>Прогрев, горячая реплика и масштабирование</h1>",
             f"<p class='lead'>Данные из локального Kubernetes. Backend: <code>simulated</code>. "
             f"Сохранено файлов завершённых нагрузочных прогонов B: {len(b_runs)} из 12.</p>",
             f"<div class='notice {'ok' if summary['complete'] else ''}'><strong>"
             + ("Все автоматически проверяемые критерии и источник кейса подтверждены." if summary["complete"] else
                "Технические проверки пройдены; остаётся подтвердить кейс из лекции." if summary["technical_complete"] else
                "Полная приёмка пока не подтверждена. Ниже отдельно указаны фактические измерения, проверки и недостающие доказательства.")
             + "</strong></div>",
             "<p>Отчёт построен повторно из сохранённых JSON/JSONL; отсутствие файла отображается как пробел, а не заменяется вымышленным результатом. "
             "Отсутствие champion принято по условиям работы; веса модели и переобучение для этого CPU-стенда не требуются.</p>",
             "<p>Исходные условия: " + asset("course_tasks/hw11/homework11.html") + ". "
             "Репозиторий: <a href='https://github.com/IvanKaruzin/MLOps'>IvanKaruzin/MLOps</a>; точный итоговый коммит сообщает результат push. "
             "Единый журнал попыток: " + raw("requests.jsonl") + "; машиночитаемые итоги: " + raw("summary.json") + "; собственные проверки: " + raw("checks.json") + ".</p>",
             "<h2>1. Стенд и воспроизводимость</h2>",
             "<p>Продолжен FastAPI-сервис ДЗ 8–10 через отдельный режим <code>--backend simulated</code>. Старые stub/transformers API сохранены. "
             "Deployment, ConfigMap, два Service, три разных probes, CPU/RAM limits и HPA заданы Helm-чартом. "
             "Отдельный kind работает в профиле Colima <code>mlops-hw11</code>; ресурсы других профилей не используются.</p>",
             "<p>Источник имитируемого артефакта — read-only ConfigMap с файлами A/B. До объявления готовности процесс проверяет SHA-256, schema и версию; "
             "ответ сообщает фактически загруженные <code>model_version</code> и <code>artifact_sha256</code>. "
             "Request ID, session ID, Pod и видимый сервером source IP сохраняются в JSON-логах; метрики не содержат request/session ID. "
             "Авторитетная история передаётся целиком в <code>messages</code>, поэтому локальный prefix cache не является хранилищем диалога.</p>",
             "<p>Параметры CPU-имитатора: OpenSSL <code>pbkdf2_hmac('sha256', ...)</code> в ограниченном пуле потоков; "
             "hit = 75 000, miss = 700 000, chunk = 25 000 итераций. Очередь ограничена 32 ожидающими, running = 1 на Pod, queue timeout = 30 с; "
             "LRU prefix cache = 16 записей. Ключ — SHA-256 первого сообщения, иначе первых 128 символов prompt. "
             "Поток сохраняет compute slot до завершения; отменённая CPU-задача удерживает слот до физического окончания thread job. "
             "В основных B: четыре части ответа, задержка 0,2 с между выдачей частей, 2 запроса/с, 120 с arrivals (240 запланированных запросов). "
             "Фактические values/config каждого прогона являются источником настроек; они приоритетнее этого описания.</p>",
             "<p>Одна API-реплика: CPU request 250m / limit 1 core, RAM 128 / 256 MiB. Prometheus: retention 2 ч / 256 MB, "
             "том ≤512 MiB; Grafana ≤128 MiB; API tmp ≤16 MiB. Основной прогрев 20 с; отдельный drain-proof использует явно сохранённые 10 с. "
             "PreStop снимает readiness, затем 5 с обслуживает остаточный трафик распространения EndpointSlice; "
             "SIGTERM прекращает новый admission, uvicorn даёт активным ответам до 50 с внутри grace period 60 с.</p>",
             json_block(environment),
             "<p>Подтверждение ресурсов до запуска: " + raw("environment-before.json") + "; после: " + raw("environment-after.json") + "; сборка и discovery: " + raw("setup.json") + "; финальная сборка: " + raw("image-build.json") + ".</p>",
             "<p>Итоговая проверка реальных targets и загруженного дашборда: " + raw("monitoring-validation.json") + "; экспорт Grafana API: " + raw("grafana-export.json") + ".</p>",
             ("<p>Удаление собственного стенда проверено: " + raw("cleanup.json") + ". Исходный Docker context восстановлен; остальные профили Colima сохранены.</p>")
             if read(base / "cleanup.json").get("passed") else "",
             "<p>Запуск и удаление: " + asset("README.md", "инструкция README") + "; " + asset("deploy/hw11/chart", "Helm-чарт") + "; "
             + asset("deploy/hw11/Dockerfile") + "; " + asset("requirements-hw11.lock") + ".</p>",
             "<pre>make install-hw11\nmake up-hw11\nmake check-hw11\nmake experiments-hw11\nmake report-hw11\nmake down-hw11\n# Полное удаление только профиля ДЗ-11:\n.venv-hw11/bin/python scripts/hw11.py down --delete-vm</pre>",
             "<h2>2. Метод измерения и полнота</h2>",
             "<p>Клиент измеряет интервалы монотонными часами: TTFC = first_chunk − send, full = end − send; "
             "inter-chunk latency — среднее N−1 интервалов между N частями. Для одной части значение отсутствует. "
             "UTC применяется только для общей линии времени; часы разных процессов не вычитаются. "
             "Генератор open-loop следует абсолютному расписанию, создаёт новое HTTP-соединение на запрос и явно сохраняет "
             "planned/actual send, lag и unsent при переполнении клиентского backlog. Четыре клиентских Pod имеют разные IP. "
             "Каждая попытка имеет свой attempt ID; повторы не объединяются с исходной попыткой.</p>",
             "<p>Успех требует HTTP 200, корректного final marker, ожидаемого количества частей и чистого окончания потока. "
             "Отказ до заголовков, обрыв, timeout, connection error и неотправленный запрос учитываются раздельно. "
             "p50/p95 приведены только для completed. Доли error/timeout/rejected делятся на sent; unsent — на planned. "
             "Клиентская полная длительность включает чтение конца ответа. Backend-гистограмма фиксирует успешное серверное завершение; небольшая сетевая разница ожидаема.</p>"]
    statistics_rows = []
    for name, data in runs.items():
        statistics_rows.append((name.removesuffix("/requests.jsonl"), data["planned"], data["sent"], data["successful"],
                                data["ttfc_seconds"]["p50"], data["ttfc_seconds"]["p95"],
                                data["full_seconds"]["p50"], data["full_seconds"]["p95"],
                                data["counts"]["rejected"], data["counts"]["interrupted"], data["counts"]["timeout"],
                                data["counts"]["connection_error"] + data["counts"]["error"], data["counts"]["unsent"]))
    parts.append(table(("Прогон", "Planned", "Sent", "N успехов", "TTFC p50, с", "TTFC p95, с", "Full p50, с", "Full p95, с", "Отказы", "Обрывы", "Timeout", "Ошибки", "Unsent"), statistics_rows))
    parts.append("<p>Единый журнал содержит " + html.escape(str(summary["aggregate"]["rows"]))
                 + " попыток из " + html.escape(str(summary["aggregate"]["source_files"]))
                 + " исходных файлов; уникальные attempt ID и терминальные исходы проверены, несовпадения входят в data_errors. "
                 "Архивы A-initial и smoke не смешиваются с основными данными. В каждой строке сохранены source_file и experiment: "
                 + raw("requests.jsonl") + ".</p>")
    parts.append("<p>Полные знаменатели, доли, lag и распределение по Pod находятся в " + raw("summary.json") + ". "
                 "Сырые попытки доступны по каталогам " + raw("a", "A") + ", " + raw("b", "B") + ", " + raw("c", "C") + ".</p>")
    parts += ["<h2>3. Опыт A — прогрев, выпуск и восстановление</h2>",
              f"<p><strong>Проверка: {status(a_checks)}.</strong> Гипотеза: HTTP-процесс может отвечать probes раньше, чем готов backend; "
              "правильный readiness/startup исключает такой Pod из рабочего Service. Исходный дефект воспроизведён отдельным <code>brokenProbes=true</code>; "
              "исправление связывает probes с проверенным артефактом и окончанием прогрева. Liveness отдельно отвечает за живость HTTP.</p>",
              "<p>Доказательства до/после: " + raw("a/faulty/warmup-assertion.json") + ", " + raw("a/fixed/warmup-assertion.json") + "; "
              "Ready/EndpointSlice и полный журнал запросов сохранены в обоих каталогах. В исправленном режиме отсутствие ранних backend-запросов "
              "проверяется отдельно от отсутствия ошибок на уже прогретых репликах.</p>",
              "<p>В исходном A→B заранее открыт длительный поток к конкретной старой реплике, а основной поток запросов идёт через Service. "
              "Границы длительного ответа и начало upgrade записаны в " + raw("a/rollout/long-stream.json") + ". "
              "Этот поток пересёк начало обновления и завершился без обрыва, но его Pod был удалён уже после окончания ответа: "
              "этот прогон доказывает пересечение rollout, а не удержание активного потока при завершении процесса. "
              "Отдельный drain-proof ниже закрывает это ограничение без изменения backend и измерений B/C. "
              "Намеренно неготовая B проверяется отдельно: " + raw("a/bad-rollout") + "; "
              "после управляемого rollback фактическая A/хеш проверены в " + raw("a/rollback/requests.jsonl") + ".</p>",
              "<p><strong>Дополнительная проверка graceful termination:</strong> до upgrade открываются 28-секундные потоки на обеих A-репликах. "
              "При том же backend прогрев новых Pod задан 10 с, чтобы удаление старой реплики пересекло активный ответ. "
              "Сохраняются два длительных ответа, 60 фоновых Service-попыток, Ready/EndpointSlice и время удаления Pod. "
              "Требование — все попытки completed без 5xx, а хотя бы один старый поток заканчивается позже deletion + preStop 5 с + запас 1 с "
              "и раньше окончания grace period 60 с. Начало завершения вычисляется как Kubernetes deletionTimestamp минус deletionGracePeriodSeconds "
              "и сверяется с первым наблюдением terminating Pod. Сам raw deletionTimestamp — дедлайн удаления, а не начало завершения. "
              "Время начала + preStop — оценка ожидаемого SIGTERM по конфигурации, "
              "а не отдельное измерение доставки сигнала. Доказательства: " + raw("a/drain-proof/proof.json") + ", "
              + raw("a/drain-proof/requests.jsonl") + ", " + raw("a/drain-proof/background/requests.jsonl") + ".</p>",
              json_block(drain_proof) if drain_proof else "<p class='notice'>Дополнительный drain-proof ещё не подтверждён.</p>",
              "<p>Принудительный SIGKILL выполняется через containerd на kind node, а не игнорируемым сигналом namespace PID1 изнутри контейнера. "
              "Сохранены оборванная попытка, идентичность контейнера до/после и восстановление: " + raw("a/forced-kill") + ". "
              "Отдельная замена Pod и продолжение двух различимых диалогов: " + raw("a/replacement") + ". "
              "Ограничение: штатный rollout без ошибок подтверждён только для зафиксированных нагрузки, продолжительности ответа и тайм-аутов.</p>",
              "<figure><img src='figures/01-warmup-rollout.png' alt='Готовность Pod и EndpointSlice во время прогрева и обновления'>"
              "<figcaption>1. Верхние панели: faulty/fixed probes с прогревом 20 с. При наличии успешного drain-proof нижняя панель показывает "
              "два активных потока и удаление Pod в отдельном прогоне с прогревом 10 с; иначе показан исходный rollout. "
              "Пунктир начало + preStop обозначает ожидаемую границу SIGTERM; raw deletionTimestamp является дедлайном, а не этой границей. "
              "Время выровнено по UTC; длительности клиентов рассчитаны монотонно.</figcaption></figure>",
              "<h2>4. Опыт B — привязка, очередь и локальный кеш</h2>",
              f"<p><strong>Проверка: {status(b_checks)}; найдено {len(b_runs)}/12 основных прогонов.</strong> Гипотеза: ClientIP сохраняет локальность prefix cache, "
              "но при 80% запросов от одного клиента перегружает закреплённый Pod, когда другая реплика имеет свободную ёмкость. "
              "Сравниваются Service с ClientIP и без affinity над той же парой Pod; HPA выключен, keep-alive выключен, кеш перед каждым run пуст. "
              "При равномерном профиле четыре клиента делят отправки поровну; при skewed один даёт 80%. "
              "Workload hashes, запросы, длины контекста, лимиты ответа и расписания совпадают между политиками.</p>",
              "<p>Сеть, IP и Service settings: " + raw("b/network.json") + ". source_ip в серверных request logs позволяет сверить фактический адрес, "
              "видимый репликой, с Pod IP генератора. Причину маршрутизации проверяем по traces, а не предполагаем round-robin на каждом HTTP-запросе.</p>",
              "<p>Прямое доказательство адреса, видимого самому механизму ClientIP на node: " + raw("b/affinity-source-ip.json") + "; "
              "сохранённая NAT-таблица: " + raw("b/iptables-nat.txt") + ". Записи <code>/proc/net/xt_recent/KUBE-SEP-*</code> "
              "сопоставляются с client Pod IP из network/config, а правила DNAT — с IP действительных API Pod. "
              "Это дополняет source_ip в HTTP-логах и позволяет отделить адрес affinity-механизма от предположений о NAT.</p>"]
    network = read(base / "b/network.json")
    kube_proxy = network.get("kube_proxy", "")
    network_lines = [line.strip() for line in kube_proxy.splitlines()
                     if any(key in line for key in ("mode:", "clusterCIDR:", "masqueradeAll:"))]
    if network_lines:
        parts.append("<p>Зафиксированные настройки kube-proxy: <code>" + html.escape("; ".join(network_lines)) +
                     "</code>. При новых соединениях обычный Service распределяет выбор backend через сетевую политику, "
                     "а ClientIP использует видимый источнику трафика адрес. Это сравнение относится к данному iptables/Pod-network стенду.</p>")
    mapping_rows, mapping_links = [], []
    for policy in ("normal", "clientip"):
        for traffic in ("uniform", "skewed"):
            selected = [run for run in b_runs if run["policy"] == policy and run["traffic"] == traffic]
            if not selected:
                continue
            selected = max(selected, key=lambda run: run["repeat"])
            trace = selected["routing"]
            for entry in trace["mapping"]:
                mapping_rows.append((policy, traffic, selected["repeat"], entry["client"], entry["client_ip"],
                                     entry["pod"], entry["requests"], ", ".join(entry["source_ips"]) or "нет сохранённого log match",
                                     entry["matched_request_ids"], entry["source_mismatch_ids"]))
            mapping_links.append(raw(selected["path"] + "/requests.jsonl", selected["path"] + " requests") + "; " +
                                 "; ".join(raw(selected["path"] + "/" + filename, filename) for filename in trace["log_files"]))
    parts.append("<h3>От клиентского IP к Pod: проверка трассой</h3><p>Для каждой комбинации ниже показан последний сохранённый repeat "
                 "(в завершённой работе — третий). Количество запросов берётся из клиентского журнала; наблюдённый source_ip — только "
                 "из серверных request_received/request_finished с тем же request_id. Отсутствующие финальные log samples не восстанавливаются догадками. "
                 "Mismatch означает, что наблюдённый набор source IP отличается от клиентского Pod IP.</p>")
    parts.append(table(("Политика", "Профиль", "Repeat", "Клиент", "Pod IP клиента", "Backend Pod", "Запросов", "source_ip сервера", "Сопоставлено ID", "Mismatch ID"), mapping_rows))
    parts.append("<p>Прямые доказательства mapping: " + "<br>".join(mapping_links) + ".</p>")
    policy_rows = []
    for traffic in ("uniform", "skewed"):
        for policy in ("normal", "clientip"):
            selected = [run for run in b_runs if run["policy"] == policy and run["traffic"] == traffic]
            ttfc = [run["summary"]["ttfc_seconds"]["p95"] for run in selected if run["summary"]["ttfc_seconds"]["p95"] is not None]
            full = [run["summary"]["full_seconds"]["p95"] for run in selected if run["summary"]["full_seconds"]["p95"] is not None]
            def measured_range(values):
                return number(percentile(values, .5)) + " [" + number(min(values) if values else None) + " … " + number(max(values) if values else None) + "]"
            total_sent = sum(run["summary"]["sent"] for run in selected)
            total_success = sum(run["summary"]["successful"] for run in selected)
            hits = sum(run["summary"]["cache"]["hits"] for run in selected)
            misses = sum(run["summary"]["cache"]["misses"] for run in selected)
            policy_rows.append((traffic, policy, len(selected), measured_range(ttfc), measured_range(full),
                                f"{total_success}/{total_sent}",
                                sum(run["summary"]["counts"]["rejected"] for run in selected),
                                sum(run["summary"]["counts"][outcome] for run in selected for outcome in ("interrupted", "connection_error", "error")),
                                sum(run["summary"]["counts"]["timeout"] for run in selected),
                                f"{hits}/{misses}; hit rate=" + number(hits/(hits+misses) if hits+misses else None)))
    parts.append(table(("Трафик", "Политика", "Повторов", "Run p95 TTFC, с: медиана [min…max]", "Run p95 full, с: медиана [min…max]", "Успех / sent", "Отказы", "Обрывы + ошибки", "Timeout", "Cache hits/misses; доля hit"), policy_rows))
    uniform_affinity = [run for run in b_runs if run["policy"] == "clientip" and run["traffic"] == "uniform"]
    if uniform_affinity:
        selected = max(uniform_affinity, key=lambda run: run["repeat"])
        counts = {name: pod["attempts"] for name, pod in selected["summary"]["per_pod"].items()}
        parts.append("<p>Отдельно сохраняем случай ClientIP + uniform: " + raw(selected["path"] + "/requests.jsonl", selected["path"]) +
                     "; запросы по Pod: " + html.escape(json.dumps(counts, ensure_ascii=False)) +
                     "; p95 TTFC этого run = " + number(selected["summary"]["ttfc_seconds"]["p95"]) + " с, "
                     "успешных = " + str(selected["summary"]["successful"]) + "/" + str(selected["summary"]["sent"]) + ". "
                     "При закреплении четырёх IP по схеме 2+2 равномерный профиль остаётся сбалансированным; локальность кеша "
                     "и фазовое расписание arrivals могут уменьшить очередь и задержку относительно случайного выбора backend на новых соединениях. "
                     "Это объяснение проверяется распределением и кеш-промахами; отдельно разделить влияние локальности и фаз без дополнительного опыта нельзя. "
                     "Этот результат учитывается наравне с горячим профилем; "
                     "обычный Service не объявляется лучшим при любой нагрузке.</p>")
    balanced = [run for run in b_runs if run["policy"] == "normal" and run["traffic"] == "skewed" and run["summary"]["per_pod"]]
    if balanced:
        chosen = min(balanced, key=lambda run: max(pod["attempts"] for pod in run["summary"]["per_pod"].values()) / max(1, run["summary"]["sent"]))
        counts = {name: pod["attempts"] for name, pod in chosen["summary"]["per_pod"].items()}
        max_share = max(counts.values()) / max(1, chosen["summary"]["sent"])
        claim = "Сохранённый результат без ожидаемого 80%-перекоса распределения" if max_share <= .7 else "Ослабление ожидаемого перекоса распределения пока не подтверждено"
        parts.append("<p><strong>" + claim + ":</strong> "
                     + raw(chosen["path"] + "/requests.jsonl", chosen["path"]) + "; фактические запросы по Pod: "
                     + html.escape(json.dumps(counts, ensure_ascii=False)) + ". При том же 80%-клиенте отсутствие ClientIP "
                     "и новые соединения допускают распределение на обе реплики; степень баланса оценивается этими числами, а не обещается сетевой реализацией.</p>")
    else:
        parts.append("<p class='notice'>Основной trace без сильного перекоса распределения ещё не сохранён; вывод о нём пока не сделан.</p>")
    decision = summary["routing_decision"]
    if decision["ready"]:
        measured = decision["heavy_profile"]
        parts.append("<p><strong>Выбор по тяжёлому профилю:</strong> медиана p95 TTFC трёх runs: normal = " +
                     number(measured["normal"]["median_run_p95_ttfc_seconds"]) + " с, ClientIP = " +
                     number(measured["clientip"]["median_run_p95_ttfc_seconds"]) + " с; доля неуспехов/sent: " +
                     number(measured["normal"]["unsuccessful_fraction_of_sent"]) + " и " +
                     number(measured["clientip"]["unsuccessful_fraction_of_sent"]) +
                     ". Промахи кеша среди completed: " + str(measured["normal"]["cache_misses_completed"]) + " и " +
                     str(measured["clientip"]["cache_misses_completed"]) + ". " +
                     ("Обычный Service подтверждён как исправление горячего профиля: задержка ниже без роста доли неуспехов." if decision["supported_fix"] == "normal" else
                      "Предложенное исправление обычным Service не подтверждено выбранным критерием; требуется разбор фактических результатов.") +
                     " Три повтора описывают этот стенд, но не дают универсальной гарантии для других сетей и workloads.</p>")
    else:
        parts.append("<p>Окончательный выбор политики по тяжёлому профилю будет сделан после всех трёх повторов обеих политик; "
                     "частичная таблица не считается доказательством преимущества.</p>")
    parts += ["<p>Диагностика " + raw("b/no-skew") + " использует ClientIP и низкую частоту: она проверяет отсутствие <em>очереди</em> ниже ёмкости; "
              "это не доказательство равномерного распределения. Калибровка " + raw("b/calibration/calibration-decision.json") + " требует одновременных waiting&gt;0 у одной реплики "
              "и running=0 у другой перед запуском 12 основных прогонов.</p>",
              "<p>Проверяемое исправление — обычный Service без постоянной привязки диалога к Pod. Его измеренный эффект оценивается отдельно для каждого профиля по таблице и трём repeats. "
              "В таблице hits/misses считаются по completed с известным cache_hit, hit rate = hits/(hits+misses); отказавшие запросы без lookup не включаются в этот знаменатель. "
              "Число дополнительных промахов при обычном Service сравнивается на том же расписании: оно показывает измеренную цену копирования префикса на другую реплику, но не размер реального KV-кеша. "
              "Цена решения — возможный пересчёт префикса на второй реплике; истории диалогов это не разрушает, потому что полный контекст находится в запросе. "
              "Числа относятся к искусственным hit/miss costs. На настоящем engine отдельно проверяются prefill/decode, batch admission, eviction, KV-память и cache-aware routing.</p>",
              "<figure><img src='figures/02-hot-replica.png' alt='Задержка, очередь, running и waiting по репликам'>"
              "<figcaption>2. Успешные запросы по Pod и сохранённые gauges Prometheus; конкретный run указан в заголовке. Наличие очереди у горячей реплики проверяется вместе со свободной второй.</figcaption></figure>",
              "<figure><img src='figures/03-policy-comparison.png' alt='Сравнение политик в трёх повторах для двух профилей'>"
              "<figcaption>3. Каждая точка — отдельный run. p95 считаются по успешным ответам; доля неуспехов имеет знаменатель sent. N рядом с точками — число успехов.</figcaption></figure>",
              "<h3>Сравнение classic histogram с тем же набором ответов</h3>",
              "<p>Для каждого B run сохранены cumulative buckets до arrivals и после завершения всех запросов, без смены Pod и чужого workload. "
              "Разность bucket-счётчиков суммируется по <code>le</code>; проверяются неизменность пары Pod, отсутствие counter reset, "
              "совпадение delta_count и N completed клиента. p50/p95 гистограммы используют линейную интерполяцию внутри bucket, "
              "точные серверные квантили — поля done, точные клиентские — monotonic durations. Разница сети и аппроксимации не скрывается.</p>",
              "<p>PromQL, sliding-window оговорка и alert expressions: " + asset("deploy/hw11/monitoring/promql.md") + "; "
              + asset("deploy/hw11/monitoring/alerts.yml") + ". Перцентиль рассчитывается после агрегации с сохранением <code>le</code>, например:</p>",
              "<pre>histogram_quantile(0.95,\n  sum by (le, pod) (rate(hw11_ttfc_seconds_bucket[1m])))</pre>"]
    hist_rows = []
    for run in b_runs:
        for key in ("ttfc", "full"):
            item = run["cohort"].get(key, {})
            hist_rows.append((run["path"], key, item.get("valid", False), item.get("delta_count"),
                              item.get("client", {}).get("p50"), item.get("server_exact", {}).get("p50"), item.get("histogram_p50_seconds"),
                              item.get("client", {}).get("p95"), item.get("server_exact", {}).get("p95"), item.get("histogram_p95_seconds"),
                              run["cohort"].get("prometheus", {}).get(key, {}).get("valid", False),
                              run["cohort"].get("prometheus", {}).get(key, {}).get("histogram_p95_seconds")))
    parts.append(table(("Run", "Метрика", "Cohort верен", "Delta N", "Client p50, с", "Server p50, с", "Histogram p50, с", "Client p95, с", "Server p95, с", "Histogram p95, с", "Prom cohort верен", "Prom p95, с"), hist_rows))
    hot_histograms = [run["cohort"]["ttfc"] for run in b_runs
                      if run["policy"] == "clientip" and run["traffic"] == "skewed"
                      and run["cohort"].get("ttfc", {}).get("valid")]
    if hot_histograms:
        client_p95 = percentile([item["client"]["p95"] for item in hot_histograms], .5)
        server_p95 = percentile([item["server_exact"]["p95"] for item in hot_histograms], .5)
        histogram_p95 = percentile([item["histogram_p95_seconds"] for item in hot_histograms], .5)
        bounds = sorted(float(upper) for upper in hot_histograms[0]["buckets"] if finite(float(upper)))
        lower = max((upper for upper in bounds if server_p95 is not None and upper <= server_p95), default=0)
        upper = min((upper for upper in bounds if server_p95 is not None and upper >= server_p95), default=None)
        parts.append("<p class='notice'><strong>Ограничение разрешения гистограммы:</strong> в ClientIP + skewed медиана run p95 "
                     "по сырым клиентским длительностям = " + number(client_p95) + " с, по точным серверным длительностям = "
                     + number(server_p95) + " с, а интерполяция classic histogram даёт " + number(histogram_p95) + " с. "
                     "Широкий bucket " + number(lower) + "…" + number(upper) + " с распределяет наблюдения линейно внутри границ, "
                     "хотя реальные значения сосредоточены в части интервала. Значение " + number(histogram_p95)
                     + " с является оценкой гистограммы и не используется как фактически измеренный p95. "
                     "Выбор политики опирается на сырые клиентские данные и долю отказов. "
                     "В следующей версии следует добавить границы 26/28/30/32 с; текущие результаты и buckets сохранены без изменения. "
                     "Численная погрешность аппроксимации не меняет подтверждённое совпадение набора ответов и delta_count.</p>")
    hot_counts = hot_hpa.get("per_pod", [])
    zero_new = [entry for entry in hot_counts if entry["new_during_observation"] and entry["attempts"] == 0]
    hot_observation = ""
    if zero_new:
        hot_observation = ("<p><strong>Наблюдение этого прогона:</strong> " + "; ".join(
            html.escape(entry["pod"]) + ": Ready через " + number(entry.get("creation_to_ready_seconds"))
            + " с после создания, 0 из " + str(hot_hpa["sent"]) + " записанных HTTP-попыток"
            for entry in zero_new) + ". Другие наблюдённые реплики получили " + ", ".join(
                html.escape(entry["pod"]) + " — " + str(entry["attempts"])
                for entry in hot_counts if entry not in zero_new) + ". Это фактический случай, когда добавленная готовая ёмкость "
                "не разгрузила закреплённых клиентов.</p>")
    parts += ["<h2>5. Опыт C — HPA и два алерта</h2>",
              f"<p><strong>Проверка: {status(c_checks)}.</strong> Гипотеза: вычислительная нагрузка увеличит среднюю CPU-утилизацию выше target, "
              "HPA добавит третью реплику, после снижения arrivals вернёт минимум с учётом stabilization window. "
              "Backend и стоимость вычислений те же, что в B; CPU pressure меняется частотой запросов и задержкой между частями, а не другой реализацией имитатора. "
              "HPA <code>autoscaling/v2</code>: минимум 2, максимум 3, target 70% относительно CPU requests, scale-down stabilization 30 с.</p>",
              "<p>Условия HPA и доступность resource metrics API: " + raw("c/hpa-before.json") + ", " + raw("c/metrics-api.json") + "; "
              "измеренный цикл, создание/готовность Pod и scale-down: " + raw("c/hpa/timeline.jsonl") + ", " + raw("c/hpa/scale-down.json") + ". "
              "Момент снижения arrivals берётся из абсолютного расписания, а окончание drain клиентов сохраняется отдельно.</p>",
              json_block(summary["hpa"]),
              "<p>Горячий ClientIP-профиль повторён при HPA: " + raw("c/hot-replica") + ". Средняя CPU может пересечь target, если работа горячего Pod достаточна относительно суммы requests; "
              "при низкой средней CPU новая реплика не появится. Даже после scale-out привязанный IP продолжает попадать на прежний Pod, "
              "поэтому третья реплика сама по себе не гарантирует снятие очереди; это проверяется traces и running/waiting.</p>",
              "<p>Фактическое распределение горячего прогона ниже строится по идентичности Pod в HTTP-ответах и всем API Pod, "
              "наблюдённым в timeline. Поэтому новая готовая реплика с нулём записанных попыток показана явно, а не пропущена. "
              "Время создания и первого Ready берётся из Kubernetes; интервал creation → Ready включает запуск и прогрев. "
              "Источник: " + raw("c/hot-replica/requests.jsonl") + ", " + raw("c/hot-replica/timeline.jsonl") + ".</p>",
              hot_observation,
              table(("Pod", "Новый в опыте", "Попыток", "Completed", "Rejected", "Created UTC", "Ready UTC", "Creation → Ready, с"),
                    [(entry["pod"], entry["new_during_observation"], entry["attempts"], entry["successful"], entry["rejected"],
                      entry["created_utc"], entry["ready_utc"], entry.get("creation_to_ready_seconds")) for entry in hot_counts]),
              json_block(hot_hpa),
              "<p>Алерты: p95 TTFC &gt;0,8 с при ≥10 успешных наблюдениях за 30 с; доля rejected &gt;5% при ≥20 terminal outcomes за 30 с. "
              "Оба имеют <code>for: 10s</code>, evaluation interval 5 с. Реальные inactive/pending/firing/normal состояния и правила: "
              + raw("c/alerts/alert-states.json") + ", " + raw("c/alerts/rules.json") + ". Наличие YAML правила само по себе не считается срабатыванием.</p>",
              "<figure><img src='figures/04-hpa.png' alt='HPA 2-3-2, CPU и поданная нагрузка'>"
              "<figcaption>4. Фактические replicas/Ready, CPU в cores и отправки/с. Вертикальная линия — конец запланированных arrivals; Pod creation/ready timestamps сохранены рядом в JSON.</figcaption></figure>",
              "<h2>6. Собственные проверки и соответствие условиям</h2>",
              "<p>Обязательные live-проверки обращаются к Kubernetes-сервису: прогрев, фактическая загруженная A после отката, восстановление версии и диалога на заменённом Pod. "
              "Дополнительно проверяются реальные rejected/interrupted и изоляция двух диалогов с различимыми контрольными строками. "
              "Исходный сбой faulty probes сохранён отдельно от исправленного PASS. Unit tests дополняют эти доказательства, а не заменяют их.</p>",
              json_block(summary["checks"]),
              "<p>Регрессия текущего проекта: " + raw("regression.json") + "; всего проверок: "
              + html.escape(str(summary["regression"].get("total_tests", "ещё нет данных"))) + "; passed="
              + html.escape(str(summary["regression"].get("passed"))) + ".</p>", json_block(summary["regression"])]
    matrix = [
        ("Основа и ресурсы", "CPU simulated, Docker/kind, pinned deps/images, диск до/после, limits/retention/cleanup", "environment-before.json; environment-after.json; setup.json; ../../deploy/hw11; ../../README.md"),
        ("Helm и probes", "Deployment2, Service/ConfigMap, image/model/warmup/replicas values; startup/readiness/liveness раздельно", "../../deploy/hw11/chart; a/faulty; a/fixed"),
        ("Версия и диалог", "Проверка hash A/B; request/Pod/version identity; stateless history; замена Pod", "a/rollback/requests.jsonl; a/replacement"),
        ("Наблюдаемость", "Prometheus per-Pod discovery, Grafana export, CPU/RAM/Ready/running/waiting/errors/latency", "setup.json; ../../deploy/hw11/monitoring/grafana-dashboard.json; b; c"),
        ("A: исходный дефект", "Ранний unready request, Ready/EndpointSlice, затем тот же опыт с исправленными probes", "a/faulty/warmup-assertion.json; a/fixed/warmup-assertion.json"),
        ("A: rollout/rollback", "Активный поток пересекает rollout и удаление Pod после preStop; 0 5xx/unfinished; never-ready B исключена; rollback A по ответам", "a/rollout; a/drain-proof/proof.json; a/bad-rollout; a/rollback"),
        ("A: отказ", "Реальный принудительный процессный отказ, обрыв и восстановление; отдельные attempt IDs", "a/forced-kill"),
        ("B: контролируемое сравнение", "Два Service, ≥4 Pod/IP, keep-alive учтён, uniform и80%, одинаковые inputs/schedules", "b/network.json; b/normal; b/clientip"),
        ("B: кеш и горячая реплика", "Capacity/hit/miss/concurrency; waiting+idleother; результат без сильного skew с объяснением", "b/calibration; b/no-skew; b/normal/skewed; figures/02-hot-replica.png"),
        ("B: повторяемость", "2policy×2traffic×3repeats, ≥120s каждый; initialcache, planned/actualsend и lag", "b; summary.json; checks.json"),
        ("C: HPA", "metrics API, requests, actual CPU, HPA2..3, 2→3→2, creation/Ready и stabilization duration", "c/metrics-api.json; c/hpa-before.json; c/hpa"),
        ("C: hot+alerts", "ClientIP при HPA; два thresholds/minimumN/for, pending/firing/normal", "c/hot-replica; c/alerts"),
        ("Измерения", "Монотонные интервалы, chunks, terminal split, N+p50/p95, ratios и bounded labels", "a; b; c; summary.json"),
        ("Гистограммы", "sum buckets с le, cohortdelta_count=N, client/server/Prometheus comparison", "b/normal; b/clientip; ../../deploy/hw11/monitoring/promql.md"),
        ("Собственные проверки", "≥5, триliveKubernetes + accounting + dialogueisolation; исходныйFAIL/исправленныйPASS", "checks.json; ../../scripts/hw11_checks.py; ../../tests_hw11"),
        ("Сдача и защита", "Четыре графика, HTML, raw journals, команды, PR review + подтверждённый кейс лекции/первоисточник", "figures; report.html; lecture-case.json"),
    ]
    monitoring = read(base / "monitoring-validation.json")
    stand = {"passed": bool(before and after and read(base / "image-build.json")
                            and monitoring.get("passed"))} if after and monitoring else {}
    evidence_status = {
        "Основа и ресурсы": stand, "Helm и probes": a_checks,
        "Версия и диалог": a_checks, "Наблюдаемость": monitoring,
        "A: исходный дефект": a_checks, "A: rollout/rollback": a_checks, "A: отказ": a_checks,
        "B: контролируемое сравнение": b_checks, "B: кеш и горячая реплика": b_checks,
        "B: повторяемость": b_checks, "C: HPA": c_checks, "C: hot+alerts": c_checks,
        "Измерения": {"passed": summary["technical_complete"]},
        "Гистограммы": {"passed": len(b_runs) == 12 and not summary["histogram_cohort_errors"]},
        "Собственные проверки": {"passed": all(x.get("passed") for x in (a_checks, b_checks, c_checks))},
        "Сдача и защита": {"passed": summary["complete"]},
    }
    parts.append("<div class='table'><table><thead><tr><th>Группа критериев</th><th>Статус</th><th>Что проверяется</th><th>Прямые доказательства</th></tr></thead><tbody>" + "".join(
        "<tr><td>" + html.escape(name) + "</td><td>" + status(evidence_status[name]) + "</td><td>" + html.escape(description) + "</td><td>" + "; ".join(
            link(base, base / target.strip()) for target in evidence.split(";")) + "</td></tr>" for name, description, evidence in matrix) + "</tbody></table></div>")
    parts += ["<h2>7. Ревью предложения закрепить диалог навсегда</h2>",
              "<p>При очереди на A в 3 секунды и пересчёте префикса на свободной B за 0,8 секунды новый запрос следует допустить на B, "
              "если остальные затраты и ограничения сопоставимы. Разница 2,2 секунды до начала полезной работы больше выигрыша от локального кеша на A. "
              "Постоянное закрепление диалога за Pod превращает полезную локальность в ограничение доступной ёмкости; кеш должен влиять на стоимость маршрута, "
              "а не быть единственным критерием.</p>",
              "<p>Перед production-решением нужны распределения queue/prefill/decode, точность прогноза cache hit, размер и eviction KV-кеша, "
              "стоимость передачи состояния, свободная GPU/CPU-память, batch admission и p95/p99 на одинаковом workload. "
              "Новый запрос допускается только при наличии capacity и укладывающемся в deadline прогнозе; иначе нужен ограниченный отказ или "
              "маршрут на другую допустимую реплику. Решение проверяется по задержкам, отказам и распределению работы, а не средней загрузке.</p>",
              "<p>История диалога должна жить у клиента или во внешнем хранилище. После потери A B восстанавливает её из полного контекста, "
              "а локальный префикс пересчитывает. Уже начатый поток остаётся на обслуживающем engine до завершения в grace period; "
              "перенос посередине ответа допускается только при явной поддержке engine и протокола. После аварийного обрыва новая попытка "
              "получает отдельный attempt ID; повторная генерация не выдаётся за бесшовное продолжение старого потока.</p>"]
    if summary["lecture_source_confirmed"]:
        parts.append("<p>Подтверждённый кейс лекции: <strong>" + html.escape(lecture["company"]) + "</strong>. "
                     + html.escape(lecture["case_text"]) + " "
                     + f'<a href="{html.escape(lecture["lecture_url"], quote=True)}">Лекция</a>; '
                     + f'<a href="{html.escape(lecture["primary_url"], quote=True)}">первичный источник</a>. '
                     + raw("lecture-case.json", "Сохранённая привязка источника") + ".</p>")
    else:
        parts.append("<div class='notice'><strong>Недостающий критерий:</strong> локальная лекция 11 отсутствует, а источник её кейса ещё не предоставлен. "
                     "Поэтому это ревью пока не связано с подтверждённым кейсом компании из лекции. Произвольный внешний пример не выдан за такой кейс. "
                     "Для закрытия нужен источник лекции и первичная публикация выбранной компании.</div>")
    if summary["invariant_violations"] or summary["histogram_cohort_errors"] or summary["data_errors"] or summary["missing_required_files"]:
        parts.append("<h3>Обнаруженные ограничения данных</h3>" + json_block(dict(missing_required_files=summary["missing_required_files"], data_errors=summary["data_errors"], invariants=summary["invariant_violations"], histogram_cohorts=summary["histogram_cohort_errors"])))
    archives = [p for p in sorted(base.iterdir()) if p.is_dir() and ("attempt" in p.name.lower() or "archive" in p.name.lower() or "initial" in p.name.lower())]
    if archives:
        parts.append("<p>Сохранённые предыдущие попытки (не смешиваются с основными измерениями): " + "; ".join(link(base, p) for p in archives) + ".</p>")
    parts.append("<p class='muted'>Сформировано из сохранённых данных: " + html.escape(summary["generated_at_utc"]) + ". "
                 "Скринкаст и сценарий исключены из работы по поручению пользователя.</p></main></body></html>\n")
    (base / "report.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/hw11")
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero if any acceptance evidence is missing")
    parser.add_argument("--skip-figures", action="store_true", help="Refresh evidence and HTML while preserving already rendered figures")
    args = parser.parse_args()
    summary = generate(args.results, render_figures=not args.skip_figures)
    print(json.dumps(dict(report=str(args.results / "report.html"), complete=summary["complete"],
                          b_runs=len(summary["b_runs"]), invariant_violations=summary["invariant_violations"],
                          histogram_cohort_errors=summary["histogram_cohort_errors"],
                          lecture_source_confirmed=summary["lecture_source_confirmed"]), ensure_ascii=False))
    return int(args.require_complete and not summary["complete"])


if __name__ == "__main__":
    raise SystemExit(main())
