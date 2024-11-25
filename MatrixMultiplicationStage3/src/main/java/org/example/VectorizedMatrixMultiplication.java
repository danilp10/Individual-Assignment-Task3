package org.example;

public class VectorizedMatrixMultiplication {
    public static double[][] vectorizedMatrixMultiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];

        java.util.stream.IntStream.range(0, n).parallel().forEach(i -> {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        });

        return C;
    }
}
