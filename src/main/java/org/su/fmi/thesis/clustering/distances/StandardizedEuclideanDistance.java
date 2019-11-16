package org.su.fmi.thesis.clustering.distances;

import static org.su.fmi.thesis.clustering.distances.Utils.variances;

public class StandardizedEuclideanDistance implements Distance {

    private double[] variancesVector;

    public StandardizedEuclideanDistance(double[][] vectors) {
        variancesVector = variances(vectors);
    }

    @Override
    public double distance(double[] v1, double[] v2) {
        double dist = 0;

        for (int d = 0; d < v1.length; ++d) {
            dist += (Math.pow(v1[d] - v2[d], 2) / variancesVector[d]);
        }

        return dist;
    }

    @Override
    public String displayName() {
        return "standardized-euclidean-distance";
    }
}
