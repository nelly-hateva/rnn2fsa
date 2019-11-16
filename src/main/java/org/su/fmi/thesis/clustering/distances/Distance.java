package org.su.fmi.thesis.clustering.distances;

public interface Distance {
    double distance(double[] v1, double[] v2);
    String displayName();
}
