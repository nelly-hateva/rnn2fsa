package org.su.fmi.thesis.experiments;

import org.su.fmi.thesis.clustering.KMeans;
import org.su.fmi.thesis.clustering.distances.Distance;
import org.su.fmi.thesis.clustering.distances.EuclideanDistance;
import org.su.fmi.thesis.clustering.distances.MahalanobisDistance;
import org.su.fmi.thesis.clustering.distances.StandardizedEuclideanDistance;
import org.su.fmi.thesis.clustering.model.Vectors;
import org.su.fmi.thesis.experiments.ioutils.VectorsReader;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

import static org.su.fmi.thesis.clustering.distances.Utils.covarianceMatrix;
import static org.su.fmi.thesis.clustering.distances.Utils.inverseMatrix;
import static org.su.fmi.thesis.clustering.distances.Utils.nonDegenerate;

public class KMeansMain {

    public static void main(String[] args) throws Exception {

        if (args.length != 5) {
            System.out.println("Usage: <input-file> N K D <ED|SED|M>");
            System.exit(1);
        }

        File file = new File(args[0]);

        int N = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int D = Integer.parseInt(args[3]);

        String dist = args[4];
        if (!dist.equals("ED") && !dist.equals("SED") && !dist.equals("M")) {
            System.out.println(dist + " is not a valid distance. Select from " +
                    "ED (Euclidean Distance) or SED (Standardized Euclidean Distance) or M (Mahalanobis)");
            System.exit(1);
        }

        Vectors vectors = VectorsReader.parseVectors(file, N, D);

        Distance distance;
        if (dist.equals("ED")) {
            distance = new EuclideanDistance();
        } else if (dist.equals("SED")) {
            distance = new StandardizedEuclideanDistance(vectors.vectors);
        } else {
            double[][] covMatrix = covarianceMatrix(vectors.vectors);
            if (!nonDegenerate(covMatrix)) {
                System.out.println("WARNING: Covariance matrix is degenerate. " +
                        "Falling back to Standardized Euclidean Distance");
                distance = new StandardizedEuclideanDistance(vectors.vectors);
            } else {
                distance = new MahalanobisDistance(inverseMatrix(covMatrix));
            }
        }

        clustering(vectors, N, D, K, file, distance);
    }

    private static void clustering(Vectors vectors, int N, int D, int K, File f, Distance distance)
            throws IOException {
        System.out.println("N = " + N + " D = " + D + " K = " + K);

        KMeans kMeans = new KMeans(vectors, K, 666L, distance);
        kMeans.fit();

        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                new File(f.getParent(), String.format(f.getName() + ".assignments.txt", K))
        )))) {
            for (int i = 0; i < N; i++) {
                bw.write(String.valueOf(kMeans.clusters[i]));
                bw.newLine();
            }
        }

        try (BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(
                new File(f.getParent(), String.format(f.getName() + ".centroids.txt", K))
        )))) {
            for (int k = 0; k < K; k++) {
                for (int d = 0; d < D; d++) {
                    bw.write(String.valueOf(kMeans.centroids[k][d]));
                    bw.write(" ");
                }
                bw.newLine();
            }
        }
    }
}
