package org.su.fmi.thesis.clustering;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.su.fmi.thesis.clustering.distances.EuclideanDistance;
import org.su.fmi.thesis.clustering.model.Vectors;

class KMeansTest {

    private static void assertCentroidsEquals(double[][] expected, double[][] actual) {

        assertEquals(expected.length, actual.length);

        for (double[] centroid1 : expected) {

            boolean found = false;
            for (double[] centroid2 : actual) {
                if (Arrays.equals(centroid1, centroid2)) {
                    found = true;
                    break;
                }
            }

            assertTrue(found);
        }
    }

    @Test
    void test0() {
        // if N == K, then the centroids should be the N points
        int N = 1024;
        int D = 100;

        int[] weights = new int[N];
        double[][] vectors = new double[N][D];

        Random random = new Random(666L);
        Arrays.fill(weights, 1);
        for (int i = 0; i < vectors.length; ++i) {
            for (int j = 0; j < vectors[i].length; ++j) {
                vectors[i][j] = random.nextDouble();
            }
        }

        KMeans kMeans = new KMeans(new Vectors(vectors, weights), N, new EuclideanDistance());
        kMeans.fit();

        assertCentroidsEquals(vectors, kMeans.centroids);
    }

    @Test
    void test1() {
        // compare results with this example https://datatofish.com/k-means-clustering-python/
        double[][] vectors = {
                {25, 79}, {34, 51}, {22, 53}, {27, 78}, {33, 59}, {33, 74}, {31, 73}, {22, 57}, {35, 69}, {34, 75},
                {67, 51}, {54, 32}, {57, 40}, {43, 47}, {50, 53}, {57, 36}, {59, 35}, {52, 58}, {65, 59}, {47, 50},
                {49, 25}, {48, 20}, {35, 14}, {33, 12}, {44, 20}, {45, 5}, {38, 29}, {43, 27}, {51, 8}, {46, 7}
        };
        int N = vectors.length;

        int[] weights = new int[N];
        Arrays.fill(weights, 1);

        KMeans kMeans = new KMeans(new Vectors(vectors, weights), 3, new EuclideanDistance());
        kMeans.fit();

        double[][] expectedCentroids = new double[][]{
                {43.2d, 16.7d},
                {55.1d, 46.1d},
                {29.6d, 66.8d}
        };
        assertCentroidsEquals(expectedCentroids, kMeans.centroids);
    }

    @Test
    void test2() {
        // compare results with this example https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        double[][] vectors = {{1, 2}, {1, 4}, {1, 0}, {10, 2}, {10, 4}, {10, 0}};
        int N = vectors.length;

        int[] weights = new int[N];
        Arrays.fill(weights, 1);

        KMeans kMeans = new KMeans(new Vectors(vectors, weights), 2, new EuclideanDistance());
        kMeans.fit();

        double[][] expectedCentroids = new double[][]{
                {10.0f, 2.0f},
                {1.0f, 2.0f}
        };
        assertCentroidsEquals(expectedCentroids, kMeans.centroids);
    }
}
