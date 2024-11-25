package org.example;

import java.util.concurrent.*;

public class PararellMatrixMultiplication {
    public static double[][] parallelMatrixMultiply(double[][] A, double[][] B) throws InterruptedException, ExecutionException {
        int n = A.length;
        double[][] C = new double[n][n];
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Future<?>[] futures = new Future<?>[n];

        for (int i = 0; i < n; i++) {
            final int row = i;
            futures[i] = executor.submit(() -> {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < n; k++) {
                        C[row][j] += A[row][k] * B[k][j];
                    }
                }
            });
        }

        for (Future<?> future : futures) {
            future.get();
        }

        executor.shutdown();
        return C;
    }

}
