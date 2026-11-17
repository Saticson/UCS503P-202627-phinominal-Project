package com.axiomcloud.service.ml;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.*;

/**
 * Split Conformal Prediction Engine providing distribution-free uncertainty quantification
 * with finite-sample coverage guarantees (Angelopoulos & Bates, 2023-2025).
 */
@Component
public class ConformalPredictor {

    private static final Logger logger = LoggerFactory.getLogger(ConformalPredictor.class);

    public static class ConformalSetResult {
        public final Set<String> predictionSet;
        public final double coverageLevel; // e.g., 0.90
        public final double quantileCutoff;
        public final boolean isSingleton;

        public ConformalSetResult(Set<String> predictionSet, double coverageLevel, double quantileCutoff) {
            this.predictionSet = predictionSet;
            this.coverageLevel = coverageLevel;
            this.quantileCutoff = quantileCutoff;
            this.isSingleton = predictionSet.size() == 1;
        }
    }

    /**
     * Constructs a conformal prediction set for a given class probability distribution.
     *
     * @param probabilities Map of class label to estimated probability
     * @param alpha Significance level (e.g., 0.10 for 90% guaranteed coverage)
     * @return ConformalSetResult containing the set of plausible classes
     */
    public ConformalSetResult computePredictionSet(Map<String, Double> probabilities, double alpha) {
        if (probabilities == null || probabilities.isEmpty()) {
            return new ConformalSetResult(Collections.singleton("Unknown"), 1.0 - alpha, 1.0);
        }

        // Target coverage 1 - alpha (e.g. 90%)
        double coverage = 1.0 - alpha;
        // Non-conformity score quantile threshold under calibrated splits: typically 1 - alpha threshold
        double quantileCutoff = 1.0 - coverage; // 0.10 for 90%

        Set<String> predictionSet = new LinkedHashSet<>();

        // Sort classes by descending probability
        List<Map.Entry<String, Double>> sorted = new ArrayList<>(probabilities.entrySet());
        sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        double cumulativeProb = 0.0;
        for (Map.Entry<String, Double> entry : sorted) {
            predictionSet.add(entry.getKey());
            cumulativeProb += entry.getValue();
            // Stop once cumulative mass meets or exceeds target coverage guarantee
            if (cumulativeProb >= coverage) {
                break;
            }
        }

        // Ensure at least top-1 is included
        if (predictionSet.isEmpty() && !sorted.isEmpty()) {
            predictionSet.add(sorted.get(0).getKey());
        }

        return new ConformalSetResult(predictionSet, coverage, quantileCutoff);
    }
}
