package org.su.fmi.thesis.clustering;

import org.su.fmi.thesis.clustering.distances.Distance;
import org.su.fmi.thesis.clustering.model.Vectors;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

public class KMeans {

    public double[][] centroids;
    public int[] clusters;

    private Random random;

    private int N;
    private int K;
    private int D;

    private Vectors data;

    private Distance distance;

    KMeans(Vectors data, int K, Distance distance) {
        this(data, K, 666L, distance);
    }

    public KMeans(Vectors data, int K, long seed, Distance distance) {
        this.data = data;
        this.K = K;
        this.distance = distance;

        N = data.weights.length;
        D = data.vectors[0].length;

        centroids = new double[K][D];
        clusters = new int[N];
        random = new Random(seed);
    }

    public void fit() {
        long start = System.currentTimeMillis();

        // Initialize K centroids to random K vectors
        long t0 = System.currentTimeMillis();

        Set<Integer> randomIndices = new HashSet<>();
        while (randomIndices.size() < K) {
            randomIndices.add(random.nextInt(N));
        }

        // TODO initialize k++ first point at random the next as far as possible
        List<Integer> randomIndexes = new ArrayList<>(randomIndices);
        IntStream.range(0, K).parallel().forEach(
                j -> System.arraycopy(data.vectors[randomIndexes.get(j)], 0, centroids[j], 0, D)
        );
        System.out.println(
                "Finished centroids initialization in " + (System.currentTimeMillis() - t0)
                        + " milliseconds"
        );

        t0 = System.currentTimeMillis();
        List<Set<Integer>> clusterAssignments = assignment();
        System.out.println(
                "Finished initial assignment in " + (System.currentTimeMillis() - t0) + " milliseconds"
        );

        while (true) {
            long t1 = System.currentTimeMillis();

            List<Set<Integer>> oldClusterAssignments = new ArrayList<>(clusterAssignments);

            t0 = System.currentTimeMillis();
            updateCentroids(clusterAssignments);
            System.out.println(
                    "Finished centroids update in " + (System.currentTimeMillis() - t0) + " milliseconds"
            );

            t0 = System.currentTimeMillis();
            clusterAssignments = assignment();
            System.out.println(
                    "Finished assignment in " + (System.currentTimeMillis() - t0) + " milliseconds"
            );

            int changes = 0;
            boolean equals = true;
            for (int j = 0; j < oldClusterAssignments.size(); j++) {
                for (Integer i : oldClusterAssignments.get(j)) {
                    if (!clusterAssignments.get(j).contains(i)) {
                        changes++;
                        equals = false;
                    }
                }
            }
            if (equals) {
                break;
            } else {
                System.out.println(
                        changes + " changes in " + (System.currentTimeMillis() - t1) + " milliseconds");
            }
        }

        System.out.println(
                "Finished K Means in " + (System.currentTimeMillis() - start) + " milliseconds"
        );

        for (int j = 0; j < clusterAssignments.size(); j++) {
            for (Integer i : clusterAssignments.get(j)) {
                clusters[i] = j;
            }
        }
    }

    private List<Set<Integer>> assignment() {
        List<Set<Integer>> clusterAssignments = new ArrayList<>(K);
        for (int i = 0; i < K; i++) {
            clusterAssignments.add(Collections.synchronizedSet(new HashSet<>()));
        }

        IntStream.range(0, N).parallel().forEach(
                i -> {
                    double minDist = Double.MAX_VALUE;
                    int cluster = -1;

                    for (int j = 0; j < K; j++) {
                        double dist = distance.distance(data.vectors[i], centroids[j]);

                        if (dist < minDist) {
                            minDist = dist;
                            cluster = j;
                        }
                    }

                    clusterAssignments.get(cluster).add(i);
                }
        );
        return clusterAssignments;
    }

    private void updateCentroids(List<Set<Integer>> clusterAssignments) {
        IntStream.range(0, K).parallel().forEach(
                k -> IntStream.range(0, D).parallel().forEach(
                        d -> {
                            double sum = clusterAssignments.get(k).parallelStream()
                                    .map(i -> data.vectors[i][d] * data.weights[i]).reduce(0d, Double::sum);
                            int count = clusterAssignments.get(k).parallelStream()
                                    .map(i -> data.weights[i]).reduce(0, Integer::sum);
                            centroids[k][d] = sum / (double) count;
                        }
                )
        );
    }
}
