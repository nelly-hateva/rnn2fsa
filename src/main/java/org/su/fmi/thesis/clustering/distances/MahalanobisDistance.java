package org.su.fmi.thesis.clustering.distances;

import static org.su.fmi.thesis.clustering.distances.Utils.multiply;
import static org.su.fmi.thesis.clustering.distances.Utils.subtract;
import static org.su.fmi.thesis.clustering.distances.Utils.transpose;

@SuppressWarnings("all")
public class MahalanobisDistance implements Distance {

    private double[][] inverseCovarianceMatrix;

    public MahalanobisDistance(double[][] inverseCovarianceMatrix) {
        this.inverseCovarianceMatrix = inverseCovarianceMatrix;
    }

    @Override
    public double distance(double[] x, double[] y) {

        if (x.length != inverseCovarianceMatrix.length) {
            throw new RuntimeException("Dimensions must be equals");
        }
        if (y.length != inverseCovarianceMatrix.length) {
            throw new RuntimeException("Dimensions must be equals");
        }

        double[][] s = subtract(x, y);
        double[][] d = multiply(multiply(s, inverseCovarianceMatrix), transpose(s));
        return Math.sqrt(d[0][0]);
    }

    @Override
    public String displayName() {
        return "mahalanobis-distance";
    }
}
