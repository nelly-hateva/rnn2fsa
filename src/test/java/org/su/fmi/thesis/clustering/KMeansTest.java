package org.su.fmi.thesis.clustering;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.su.fmi.thesis.clustering.distances.EuclideanDistance;
import org.su.fmi.thesis.clustering.model.Vectors;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class KMeansTest {

    @Test
    void test0() {
        // if N == K, then the centroids should be the N points
        int N = 1024;
        int D = 100;

        int[] weights = new int[N];
        double[][] vectors = new double[N][D];

        Random r = new Random(666L);
        // r.nextInt();  changes the centroids, is this expected?
        Arrays.fill(weights, 1);
        for (int i = 0; i < vectors.length; ++i) {
            for (int j = 0; j < vectors[i].length; ++j) {
                vectors[i][j] = r.nextDouble();
            }
        }
        KMeans kMeans = new KMeans(new Vectors(vectors, weights), N, new EuclideanDistance());
        kMeans.fit();

        Set<Integer> clusters = IntStream.of(kMeans.clusters).boxed().collect(Collectors.toCollection(HashSet::new));
        assertEquals(N, clusters.size());

        assertArrayEquals(vectors, kMeans.centroids);
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
        assertArrayEquals(expectedCentroids, kMeans.centroids);
    }

    @Disabled
    @Test
    void test2() {
        // compare results with this example https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        double[][] vectors = {{1, 2}, {1, 4}, {1, 0}, {10, 2}, {10, 4}, {10, 0}};
        int N = vectors.length;

        int[] weights = new int[N];
        Arrays.fill(weights, 1);

        KMeans kMeans = new KMeans(new Vectors(vectors, weights), 2, new EuclideanDistance());
        kMeans.fit();

//        float[][] expectedCentroids = new float[][]{
//                {10.0f, 2.0f},
//                {1.0f, 2.0f}
//        };
//        assertArrayEquals(expectedCentroids, kMeans.centroids);
//        for (int i = 0; i < kMeans.centroids.length; ++i) {
//            for (int j = 0; j < kMeans.centroids[i].length; ++j) {
//                System.out.print(kMeans.centroids[i][j] + " ");
//            }
//            System.out.println();
//        }
    }
}
