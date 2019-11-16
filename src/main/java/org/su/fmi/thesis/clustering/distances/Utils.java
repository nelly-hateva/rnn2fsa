package org.su.fmi.thesis.clustering.distances;

import java.util.Arrays;

public class Utils {

    static double covariance(double[][] vectors, double[] mean, int x, int y) {
        double sum = 0d;
        for (int n = 0; n < vectors.length; ++n) {
            sum += ((vectors[n][x] - mean[x]) * (vectors[n][y] - mean[y]));
        }
        return sum / (vectors.length - 1);
    }

    static double[] getMean(double[][] vectors) {
        double[] mean = new double[vectors[0].length];

        for (int x = 0; x < vectors[0].length; ++x) {
            double sum = 0d;
            for (int n = 0; n < vectors.length; ++n) {
                sum += vectors[n][x];
            }
            mean[x] = sum / vectors.length;
        }

        return mean;
    }

    static double[][] subtract(double[] x, double[] y) {
        if (x.length != y.length) {
            throw new RuntimeException("Dimensions must be equals");
        }
        double[][] d = new double[1][x.length];
        for (int i = 0; i < x.length; ++i) {
            d[0][i] = (x[i] - y[i]);
        }
        return d;
    }

    static double[][] multiply(double[][] x, double[][] y) {
        if (x[0].length != y.length) {
            throw new RuntimeException("Matrices can't be multiplied. Matrix 1 : " + x.length + "x" + x[0].length +
                    " Matrix 2 : " + y.length + "x" + y[0].length);
        }

        double[][] matrix = new double[x.length][y[0].length];
        for (int i = 0; i < x.length; ++i) {
            for (int j = 0; j < y[0].length; ++j) {
                double sum = 0d;
                for (int k = 0; k < x[0].length; ++k) {
                    sum += x[i][k] * y[k][j];
                }
                matrix[i][j] = sum;
            }
        }
        return matrix;
    }

    public static boolean nonDegenerate(double[][] matrix) {
        if (matrix.length != matrix[0].length) {
            return false;
        }

        double detMatrix = determinant(matrix);

        return !(Math.abs(detMatrix) <= Math.pow(10, -15));
    }

    public static double[][] inverseMatrix(double[][] matrix) {
        if (!nonDegenerate(matrix)) {
            throw new RuntimeException("Matrix is degenerate");
        }

        double detMatrix = determinant(matrix);

        double[][] transposedCofactorMatrix;

        if (matrix.length == 1) {
            transposedCofactorMatrix = new double[][]{{1}};
        } else {
            transposedCofactorMatrix = transpose(cofactor(matrix));
        }

        return multiplyByScalar(transposedCofactorMatrix, 1.0 / detMatrix);
    }

    static double[][] multiplyByScalar(double[][] matrix, double v) {
        double[][] multiplied = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                multiplied[i][j] = matrix[i][j] * v;
            }
        }

        return multiplied;
    }

    static double[][] transpose(double[][] matrix) {
        double[][] transposed = new double[matrix[0].length][matrix.length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }

    static double[][] cofactor(double[][] matrix) {
        double[][] cofactorMatrix = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                cofactorMatrix[i][j] = sign(i) * sign(j) * determinant(minor(matrix, i, j));
            }
        }

        return cofactorMatrix;
    }

    static double sign(int i) {
        return (i % 2 == 0) ? 1d : -1d;
    }

    static double determinant(double[][] matrix) {
        if (matrix.length != matrix[0].length) {
            throw new RuntimeException("Matrix is not squared");
        }

        if (matrix.length == 1) {
            return matrix[0][0];
        }

        if (matrix.length == 2) {
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]);
        }

        double sum = 0d;
        for (int i = 0; i < matrix[0].length; i++) {
            sum += sign(i) * matrix[0][i] * determinant(minor(matrix, 0, i));
        }

        return sum;
    }

    static double[][] minor(double[][] matrix, int excludedRow, int excludedColumn) {

        if (excludedRow < 0 || excludedRow >= matrix.length) {
            throw new RuntimeException("Invalid row " + excludedRow);
        }
        if (excludedColumn < 0 || excludedColumn >= matrix[0].length) {
            throw new RuntimeException("Invalid column " + excludedRow);
        }

        double[][] minor = new double[matrix.length - 1][matrix[0].length - 1];

        int i = -1;
        for (int row = 0; row < matrix.length; ++row) {
            if (row == excludedRow) {
                continue;
            }

            i++;
            int j = -1;
            for (int column = 0; column < matrix[0].length; ++column) {
                if (column == excludedColumn) {
                    continue;
                }
                minor[i][++j] = matrix[row][column];
            }
        }

        return minor;
    }

    public static double[][] covarianceMatrix(double[][] vectors) {
        double[] mean = getMean(vectors);

        double[][] covarianceMatrix = new double[vectors[0].length][vectors[0].length];
        for (int x = 0; x < vectors[0].length; ++x) {
            for (int y = x; y < vectors[0].length; ++y) {
                covarianceMatrix[x][y] = covarianceMatrix[y][x] = covariance(vectors, mean, x, y);
            }
        }
        return covarianceMatrix;
    }

    static double[] variances(double[][] vectors) {
        double[] mean = getMean(vectors);

        double[] variancesVector = new double[vectors[0].length];
        for (int x = 0; x < vectors[0].length; ++x) {
            variancesVector[x] = covariance(vectors, mean, x, x);
        }

        return variancesVector;
    }

    static void print(double[][] matrix) {
        for (int i = 0; i < matrix.length; ++i) {
            System.out.println(Arrays.toString(matrix[i]));
        }
    }
}
