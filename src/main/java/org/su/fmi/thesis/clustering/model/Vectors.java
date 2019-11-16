package org.su.fmi.thesis.clustering.model;

public class Vectors {
    public double[][] vectors;
    public int[] weights;

    public Vectors(double[][] vectors, int[] weights) {
        this.vectors = vectors;
        this.weights = weights;
    }
}
