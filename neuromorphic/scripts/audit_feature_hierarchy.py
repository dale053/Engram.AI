"""
Feature -> Concept hierarchy separability audit (issue #320).

ConceptSeparabilityBenchmark only measures the end result at the concept
layer: it can't tell whether that separability was inherited almost
entirely from ConceptLayer's own k-WTA bottleneck (in which case the
"hierarchy" is nominal), or whether FeatureLayer already did meaningful
work upstream (a functional hierarchy). This script builds a small network
with both layers enabled (both default to disabled — NEURO_FEATURE_N=0,
NEURO_CONCEPT_N=0 — so the hierarchy in question doesn't even exist unless
explicitly turned on), trains it, and runs FeatureHierarchyBenchmark to
measure separability at sensory, feature, and concept layers from the same
trained network and stimulus set.

This is a research/diagnostic script, not a CI regression gate: per the
issue, discovering no measurable hierarchical value would be "an important,
if uncomfortable, result" — the script reports whatever it finds and exits
0 either way, rather than treating a negative finding as a failure.

Usage:
    cd neuromorphic && uv run python scripts/audit_feature_hierarchy.py
    cd neuromorphic && uv run python scripts/audit_feature_hierarchy.py --patterns 12 --training-reps 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuromorphic.benchmarks import (  # noqa: E402
    FeatureHierarchyBenchmark,
    generate_test_patterns,
)
from neuromorphic.config import NeuromorphicConfig  # noqa: E402
from neuromorphic.network import NeuromorphicNetwork  # noqa: E402


def _build_network(feature_n: int, concept_n: int) -> NeuromorphicNetwork:
    """Small network with both FeatureLayer and ConceptLayer enabled.

    Both default to 0 (disabled) in NeuromorphicConfig.from_env(), so the
    hierarchy this issue investigates doesn't exist in the default config —
    the audit needs both explicitly sized.
    """
    cfg = NeuromorphicConfig.from_env()
    cfg.populations.brainstem = 50
    cfg.populations.reflex_arc = 30
    cfg.populations.sensory_cortex = 200
    cfg.populations.motor_cortex = 100
    cfg.populations.cerebellum = 50
    cfg.populations.association_cortex = 150
    cfg.populations.predictive_layer = 80
    cfg.populations.working_memory = 40
    cfg.populations.feature_layer = feature_n
    cfg.populations.concept_layer = concept_n
    cfg.populations.pattern_separator = 0
    cfg.populations.meta_controller = 0
    if concept_n > 0:
        cfg.concept_layer.k_winners = max(1, concept_n // 10)  # ~10% sparsity
    return NeuromorphicNetwork(cfg)


def _report(result: dict) -> int:
    if "error" in result:
        print(f"Audit could not run: {result['error']}")
        return 1

    print("=" * 78)
    print("Feature -> Concept hierarchy separability audit (issue #320)")
    print("=" * 78)
    for layer_name in ("sensory", "feature", "concept"):
        layer = result.get(layer_name)
        if layer is None:
            print(f"\n{layer_name.capitalize()}: n/a (layer disabled)")
            continue
        print(f"\n{layer_name.capitalize()} ({layer['n_neurons']} neurons):")
        print(f"  Silhouette score:       {layer['silhouette_score']:+.4f}")
        print(f"  Linear-probe accuracy:  {layer['linear_probe_accuracy']:.4f}")
        print(f"  Separation ratio:       {layer['separation_ratio']:.4f}")

    print(
        f"\nFeature vs sensory silhouette delta: {result['feature_vs_sensory_silhouette_delta']:+.4f}"
    )
    print(
        f"Feature vs sensory accuracy delta:   {result['feature_vs_sensory_accuracy_delta']:+.4f}"
    )
    if "concept_vs_feature_silhouette_delta" in result:
        print(
            f"Concept vs feature silhouette delta: "
            f"{result['concept_vs_feature_silhouette_delta']:+.4f}"
        )
        print(
            f"Concept vs feature accuracy delta:   "
            f"{result['concept_vs_feature_accuracy_delta']:+.4f}"
        )

    print()
    verdict = result["feature_layer_improves_on_sensory"]
    if verdict is None:
        print(
            f"INCONCLUSIVE: {result.get('inconclusive_reason', 'insufficient activity to measure')}. "
            "This is not evidence about hierarchy quality — it means the "
            "network never generated enough drive for FeatureLayer to fire "
            "in this run, so no separability comparison is meaningful yet. "
            "Retry with more --training-reps/--steps-per-rep or a larger "
            "--feature-n/--concept-n/--patterns before concluding anything "
            "about the FeatureLayer -> ConceptLayer hierarchy."
        )
    elif verdict:
        print(
            "VERDICT: FeatureLayer measurably improves separability over raw "
            "sensory input on both metrics — the hierarchy is functional at "
            "this boundary."
        )
    else:
        print(
            "VERDICT: FeatureLayer does NOT measurably improve separability "
            "over raw sensory input on both metrics — per issue #320, this is "
            "an important, if uncomfortable, result: the hierarchy may be "
            "nominal rather than functional at this boundary. Consider "
            "whether Invariant 5's dendritic-compartment complexity is "
            "justified here, or whether FeatureLayer needs different "
            "training/connectivity to actually contribute."
        )
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Feature->Concept hierarchy separability audit")
    parser.add_argument("--patterns", type=int, default=8, help="Number of test patterns")
    parser.add_argument("--feature-n", type=int, default=300, help="FeatureLayer population size")
    parser.add_argument("--concept-n", type=int, default=100, help="ConceptLayer population size")
    parser.add_argument("--training-reps", type=int, default=8)
    parser.add_argument("--probe-reps", type=int, default=4)
    parser.add_argument("--steps-per-rep", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    network = _build_network(args.feature_n, args.concept_n)
    rng = np.random.default_rng(args.seed)
    patterns = generate_test_patterns(args.patterns, rng)

    bench = FeatureHierarchyBenchmark(network)
    result = bench.run(
        patterns,
        training_reps=args.training_reps,
        probe_reps=args.probe_reps,
        steps_per_rep=args.steps_per_rep,
    )
    return _report(result)


if __name__ == "__main__":
    sys.exit(main())
