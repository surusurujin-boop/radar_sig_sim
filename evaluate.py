"""시뮬레이션 시나리오 × baseline × 제안 모델 통합 평가."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baselines import DTOAPriBaseline, PDWHDBSCANBaseline, PDWKMeansBaseline, ProposedModelBaseline
from src.data.scenarios import SCENARIO_BY_ID, get_scenarios
from src.evaluation import EvaluationProtocol, evaluate_all, format_results_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate radar pulse clustering methods")
    parser.add_argument(
        "--scenario-group",
        type=str,
        default="quick",
        choices=["all", "quick"],
        help="평가 시나리오 그룹",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="*",
        default=None,
        help="특정 시나리오 ID만 평가 (예: S0 S1 S4b)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model.pt",
        help="제안 모델 체크포인트 경로",
    )
    parser.add_argument(
        "--skip-proposed",
        action="store_true",
        help="제안 모델 평가 생략",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size",
    )
    return parser.parse_args()


def build_methods(args: argparse.Namespace) -> list:
    methods = [
        DTOAPriBaseline(min_cluster_size=args.min_cluster_size),
        PDWHDBSCANBaseline(min_cluster_size=args.min_cluster_size),
        PDWKMeansBaseline(),
    ]
    if not args.skip_proposed:
        ckpt = Path(args.checkpoint)
        if ckpt.exists():
            methods.append(ProposedModelBaseline(ckpt, min_cluster_size=args.min_cluster_size))
        else:
            print(f"Warning: checkpoint not found at {ckpt}, skipping Proposed model")
    return methods


def main() -> None:
    args = parse_args()

    if args.scenarios:
        scenarios = [SCENARIO_BY_ID[sid] for sid in args.scenarios]
    else:
        scenarios = get_scenarios(args.scenario_group)

    protocol = EvaluationProtocol(test_scenarios=scenarios)
    methods = build_methods(args)

    print("=" * 60)
    print("Radar Pulse Clustering Evaluation")
    print("=" * 60)
    print(f"Scenarios: {[s.scenario_id for s in scenarios]}")
    print(f"Methods:   {[m.name for m in methods]}")
    print()

    results = evaluate_all(methods, protocol.test_scenarios)

    # ARI 테이블
    print(format_results_table(results))
    print()

    # 상세 지표
    print("Detailed metrics (pooled):")
    print(f"{'Scenario':<8} {'Method':<22} {'ARI':>6} {'NMI':>6} {'Purity':>7} {'F1':>6} {'|dK|':>5} {'Noise%':>7}")
    print("-" * 75)
    for r in results:
        m = r.metrics
        print(
            f"{r.scenario_id:<8} {r.method_name:<22} "
            f"{m.ari:>6.4f} {m.nmi:>6.4f} {m.purity:>7.4f} "
            f"{m.pairwise_f1:>6.4f} {m.cluster_count_error:>5} {m.noise_ratio * 100:>6.1f}%"
        )

    # JSON 저장
    output = {
        "protocol": {
            "scenario_group": args.scenario_group,
            "scenarios": [s.scenario_id for s in scenarios],
            "metrics": ["ARI", "NMI", "Purity", "V-measure", "Pairwise-F1", "cluster_count_error"],
            "exclude_noise_in_metrics": protocol.exclude_noise_in_metrics,
        },
        "results": [
            {
                "scenario_id": r.scenario_id,
                "method": r.method_name,
                **{k: (float(v) if isinstance(v, (float, int)) else v) for k, v in r.metrics.to_dict().items()},
            }
            for r in results
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
