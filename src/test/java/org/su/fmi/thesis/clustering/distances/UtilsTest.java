package org.su.fmi.thesis.clustering.distances;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.su.fmi.thesis.clustering.distances.Utils.covarianceMatrix;
import static org.su.fmi.thesis.clustering.distances.Utils.determinant;
import static org.su.fmi.thesis.clustering.distances.Utils.inverseMatrix;
import static org.su.fmi.thesis.clustering.distances.Utils.minor;
import static org.su.fmi.thesis.clustering.distances.Utils.multiply;
import static org.su.fmi.thesis.clustering.distances.Utils.multiplyByScalar;
import static org.su.fmi.thesis.clustering.distances.Utils.sign;
import static org.su.fmi.thesis.clustering.distances.Utils.subtract;
import static org.su.fmi.thesis.clustering.distances.Utils.transpose;

class UtilsTest {

    @Test
    void test_subtract() {
        double[] v1 = new double[]{0, 1, 2, 3, 4};
        double[] v2 = new double[]{5, 6, 7, 8, 9, 10};
        double[] finalV1 = v1;
        double[] finalV2 = v2;
        Assertions.assertThrows(RuntimeException.class, () -> subtract(finalV1, finalV2));

        v1 = new double[]{0, 1, 2, 3, 4};
        v2 = new double[]{5, 6, 7, 8, 9};

        double[][] expected = new double[][]{{-5, -5, -5, -5, -5}};
        assertArrayEquals(expected, subtract(v1, v2));

        expected = new double[][]{{5, 5, 5, 5, 5}};
        assertArrayEquals(expected, subtract(v2, v1));

        expected = new double[][]{{0, 0, 0, 0, 0}};
        assertArrayEquals(expected, subtract(v1, v1));
        assertArrayEquals(expected, subtract(v2, v2));
    }

    @Test
    void test_multiply() {
        double[][] a = new double[][]{{0, 1, 2, 3, 4}};
        double[][] b = new double[][]{{5, 6, 7, 8, 9}};
        double[][] finalA = a;
        double[][] finalB = b;
        Assertions.assertThrows(RuntimeException.class, () -> multiply(finalA, finalB));

        a = new double[][]{{0, 1, 2, 3, 4}};
        b = new double[][]{{1}, {1}, {1}, {1}, {1}};

        double[][] expected = new double[][]{{10}};
        assertArrayEquals(expected, multiply(a, b));

        a = new double[][]{{0, 1}, {0, 0}};
        b = new double[][]{{0, 0}, {1, 0}};

        expected = new double[][]{{1, 0}, {0, 0}};
        assertArrayEquals(expected, multiply(a, b));

        expected = new double[][]{{0, 0}, {0, 1}};
        assertArrayEquals(expected, multiply(b, a));

        a = new double[][]{{4, 2, 4}, {8, 3, 1}};
        b = new double[][]{{3, 5}, {2, 8}, {7, 9}};

        expected = new double[][]{{44, 72}, {37, 73}};
        assertArrayEquals(expected, multiply(a, b));
    }

    @Test
    void test_inverseMatrix() {
        double[][] a = new double[][]{
                //@formatter:off
                { 666, 666 },
                //@formatter:on
        };
        double[][] finalA = a;
        Assertions.assertThrows(RuntimeException.class, () -> inverseMatrix(finalA));

        a = new double[][]{
                //@formatter:off
                { 1, 2 },
                { 2, 4 }
                //@formatter:on
        };
        double[][] finalAA = a;
        Assertions.assertThrows(RuntimeException.class, () -> inverseMatrix(finalAA));

        a = new double[][]{{1}};
        assertArrayEquals(a, inverseMatrix(a));

        a = new double[][]{
                {1, 0},
                {0, 1},
        };
        assert2dArrayEquals(a, inverseMatrix(a));

        a = new double[][]{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
        };
        assert2dArrayEquals(a, inverseMatrix(a));

        a = new double[][]{
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}
        };
        assert2dArrayEquals(a, inverseMatrix(a));

        a = new double[][]{
                //@formatter:off
                { 2,   0 },
                { 0, 100 },
                //@formatter:on
        };
        double[][] expected = new double[][]{
                //@formatter:off
                { 1 / 2d,        0 },
                {      0, 1 / 100d },
                //@formatter:on
        };
        assert2dArrayEquals(expected, inverseMatrix(a));

        a = new double[][]{{ 1024 }};
        expected = new double[][]{{ 1 / 1024d }};
        assert2dArrayEquals(expected, inverseMatrix(a));
    }

    private void assert2dArrayEquals(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; ++i) {
            assertArrayEquals(a[i], b[i], Math.pow(10, -10));
        }
    }

    @Test
    void test_multiplyByScalar() {
        double[][] a = new double[][]{{0, 1, 2, 3, 4}};
        double[][] expected = new double[][]{{0, 2, 4, 6, 8}};
        assertArrayEquals(expected, multiplyByScalar(a, 2d));

        a = new double[][]{{0, 1}, {0, 0}};
        expected = new double[][]{{0, 2}, {0, 0}};
        assertArrayEquals(expected, multiplyByScalar(a, 2d));
    }

    @Test
    void test_transpose() {
        double[][] a = new double[][]{{0, 1, 2}, {3, 4, 5}};
        double[][] expected = new double[][]{{0, 3}, {1, 4}, {2, 5}};
        assertArrayEquals(expected, transpose(a));

        a = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        expected = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        assertArrayEquals(expected, transpose(a));

        a = new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        expected = new double[][]{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}};
        assertArrayEquals(expected, transpose(a));
    }

    @Test
    void test_sign() {
        assertEquals(1, sign(0));
        assertEquals(-1, sign(1));
        assertEquals(1, sign(2));
        assertEquals(-1, sign(3));
        assertEquals(1, sign(4));
        assertEquals(-1, sign(5));
        assertEquals(1, sign(6));
    }

    @Test
    void test_determinant() {
        double[][] matrix = new double[][]{
                //@formatter:off
                { 666, 666 },
                //@formatter:on
        };
        double[][] finalMatrix = matrix;
        Assertions.assertThrows(RuntimeException.class, () -> determinant(finalMatrix));

        matrix = new double[][]{
                //@formatter:off
                { 666 },
                //@formatter:on
        };
        assertEquals(666, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { 1, 0 },
                { 0, 1 },
                //@formatter:on
        };
        assertEquals(1, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { 100, 2 },
                { 300, 6 },
                //@formatter:on
        };
        assertEquals(0, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { 3, 8 },
                { 4, 6 },
                //@formatter:on
        };
        assertEquals(-14, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { -2, 2, -3},
                { -1, 1,  3},
                {  2, 0, -1}
                //@formatter:on
        };
        assertEquals(18, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { 6,  1, 1},
                { 4, -2, 5},
                { 2,  8, 7}
                //@formatter:on
        };
        assertEquals(-306, determinant(matrix));

        matrix = new double[][]{
                //@formatter:off
                { 1, 0, 0, 0},
                { 0, 1, 0, 0},
                { 0, 0, 1, 0},
                { 0, 0, 0, 1}
                //@formatter:on
        };
        assertEquals(1, determinant(matrix));
    }

    @Test
    void test_minor() {
        double[][] matrix = new double[][]{
                //@formatter:off
                { 0,  1,  2,  3,  4},
                { 5,  6,  7,  8,  9},
                {10, 11, 12, 13, 14}
                //@formatter:on
        };

        double[][] e_00 = new double[][]{
                //@formatter:off
                { 6,  7,  8,  9},
                {11, 12, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_00, minor(matrix, 0, 0));

        double[][] e_01 = new double[][]{
                //@formatter:off
                { 5, 7,  8,  9},
                {10, 12, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_01, minor(matrix, 0, 1));

        double[][] e_02 = new double[][]{
                //@formatter:off
                { 5,  6, 8,  9},
                {10, 11, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_02, minor(matrix, 0, 2));

        double[][] e_03 = new double[][]{
                //@formatter:off
                { 5,  6,  7, 9},
                {10, 11, 12, 14}
                //@formatter:on
        };
        assertArrayEquals(e_03, minor(matrix, 0, 3));

        double[][] e_04 = new double[][]{
                //@formatter:off
                { 5,  6,  7,  8},
                {10, 11, 12, 13}
                //@formatter:on
        };
        assertArrayEquals(e_04, minor(matrix, 0, 4));

        double[][] e_10 = new double[][]{
                //@formatter:off
                { 1,  2,  3,  4},
                {11, 12, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_10, minor(matrix, 1, 0));

        double[][] e_11 = new double[][]{
                //@formatter:off
                { 0, 2,  3,  4},
                {10, 12, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_11, minor(matrix, 1, 1));

        double[][] e_12 = new double[][]{
                //@formatter:off
                { 0,  1, 3,  4},
                {10, 11, 13, 14}
                //@formatter:on
        };
        assertArrayEquals(e_12, minor(matrix, 1, 2));

        double[][] e_13 = new double[][]{
                //@formatter:off
                { 0,  1,  2, 4},
                {10, 11, 12, 14}
                //@formatter:on
        };
        assertArrayEquals(e_13, minor(matrix, 1, 3));

        double[][] e_14 = new double[][]{
                //@formatter:off
                { 0,  1,  2,  3},
                {10, 11, 12, 13}
                //@formatter:on
        };
        assertArrayEquals(e_14, minor(matrix, 1, 4));

        double[][] e_20 = new double[][]{
                //@formatter:off
                { 1,  2,  3,  4},
                { 6,  7,  8,  9}
                //@formatter:on
        };
        assertArrayEquals(e_20, minor(matrix, 2, 0));

        double[][] e_21 = new double[][]{
                //@formatter:off
                { 0, 2,  3,  4},
                { 5, 7,  8,  9}
                //@formatter:on
        };
        assertArrayEquals(e_21, minor(matrix, 2, 1));

        double[][] e_22 = new double[][]{
                //@formatter:off
                { 0,  1, 3,  4},
                { 5,  6, 8,  9}
                //@formatter:on
        };
        assertArrayEquals(e_22, minor(matrix, 2, 2));

        double[][] e_23 = new double[][]{
                //@formatter:off
                { 0,  1,  2, 4},
                { 5,  6,  7, 9}
                //@formatter:on
        };
        assertArrayEquals(e_23, minor(matrix, 2, 3));

        double[][] e_24 = new double[][]{
                //@formatter:off
                { 0,  1,  2,  3},
                { 5,  6,  7,  8}
                //@formatter:on
        };
        assertArrayEquals(e_24, minor(matrix, 2, 4));

        Assertions.assertThrows(RuntimeException.class, () -> minor(matrix, -1, 4));
        Assertions.assertThrows(RuntimeException.class, () -> minor(matrix, 3, 4));
        Assertions.assertThrows(RuntimeException.class, () -> minor(matrix, 0, -1));
        Assertions.assertThrows(RuntimeException.class, () -> minor(matrix, 0, 5));
    }

    @Test
    void test_covariance() {
        double[][] vectors = new double[][]{{1692, 68}, {1978, 102}, {1884, 110}, {2151, 112}, {2519, 154}};

        double[][] expected = new double[][]{
                //@formatter:off
                { 97732.7, 9107.3},
                {  9107.3,  941.2}
                //@formatter:on
        };
        assertArrayEquals(expected, covarianceMatrix(vectors));
    }
}
