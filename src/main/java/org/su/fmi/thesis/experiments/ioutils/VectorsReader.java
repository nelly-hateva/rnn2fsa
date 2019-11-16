package org.su.fmi.thesis.experiments.ioutils;

import org.su.fmi.thesis.clustering.model.Vectors;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class VectorsReader {

    public static Vectors parseVectors(File file, int N, int D) throws IOException {
        int n = 0;
        double[][] vectors = new double[N][D];
        int[] weights = new int[N];

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] split = line.split("\\t");
                if (split.length != 2) {
                    System.out.println("Invalid input");
                    System.exit(1);
                }
                weights[n] = Integer.parseInt(split[0]);

                split = split[1].split("\\s");

                if (split.length != D) {
                    System.out.println(
                            "WARNING. Line " + n + " vector dimension is " + split.length + ". Expected " + D
                    );
                    System.exit(2);
                }

                for (int p = 0; p < split.length; p++) {
                    vectors[n][p] = Double.parseDouble(split[p]);
                }

                n++;
            }
        }

        return new Vectors(vectors, weights);
    }
}
