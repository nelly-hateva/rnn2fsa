package org.su.fmi.thesis.clustering.distances;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.su.fmi.thesis.clustering.distances.Utils.covarianceMatrix;
import static org.su.fmi.thesis.clustering.distances.Utils.inverseMatrix;

class MahalanobisDistanceTest {

    @Test
    void test_distance() {
        double[][] vectors = new double[][]{{64, 580, 29}, {66, 570, 33}, {68, 590, 37}, {69, 660, 46}, {73, 600, 55}};

        double[][] inverseCovarianceMatrix = inverseMatrix(covarianceMatrix(vectors));
        MahalanobisDistance mahalanobis = new MahalanobisDistance(inverseCovarianceMatrix);

        double[] x = new double[]{66, 640, 44};
        double[] y = new double[]{68, 600, 40};
        assertEquals(5.33, mahalanobis.distance(x, y), 0.01);

        inverseCovarianceMatrix = new double[][]{{1, 0.5, 0.5}, {0.5, 1, 0.5}, {0.5, 0.5, 1}};
        mahalanobis = new MahalanobisDistance(inverseCovarianceMatrix);

        x = new double[]{1, 0, 0};
        y = new double[]{0, 1, 0};
        assertEquals(1, mahalanobis.distance(x, y), 0.01);

        x = new double[]{2, 0, 0};
        y = new double[]{0, 1, 0};
        assertEquals(1.73, mahalanobis.distance(x, y), 0.01);
    }
}
