package org.su.fmi.thesis.clustering.distances;

public class EuclideanDistance implements Distance {
    @Override
    public double distance(double[] v1, double[] v2) {
        double dist = 0;

        for (int d = 0; d < v1.length; ++d) {
            dist += Math.pow(v1[d] - v2[d], 2);
        }

        return dist;
    }

    @Override
    public String displayName() {
        return "euclidean-distance";
    }
}
