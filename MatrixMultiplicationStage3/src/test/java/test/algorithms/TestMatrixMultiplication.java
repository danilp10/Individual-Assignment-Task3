package test.algorithms;

import org.openjdk.jmh.annotations.*;
import com.sun.management.OperatingSystemMXBean;
import java.lang.management.ManagementFactory;
import java.util.concurrent.TimeUnit;
import java.util.Map;
import java.util.HashMap;

import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import static org.example.MatrixMultiplication.matrixMultiply;
import static org.example.VectorizedMatrixMultiplication.vectorizedMatrixMultiply;
import static org.example.PararellMatrixMultiplication.parallelMatrixMultiply;

@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(1)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class TestMatrixMultiplication {

    @Param({"100", "250", "500", "1000", "2000"})
    private int size;

    private double[][] A;
    private double[][] B;

    private Map<String, Result> results = new HashMap<>();
    private static final int NUM_THREADS = Runtime.getRuntime().availableProcessors();

    @Setup(Level.Trial)
    public void setup() {
        A = new double[size][size];
        B = new double[size][size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = Math.random();
                B[i][j] = Math.random();
            }
        }
    }

    private Result basicResult;
    private Result parallelResult;
    private Result vectorizedResult;

    @Benchmark
    public void benchmarkBasicMatrixMult() {
        basicResult = measure("Basic Matrix Multiplication", () -> matrixMultiply(A, B), 1);
    }

    @Benchmark
    public void benchmarkParallelMatrixMult() {
        parallelResult = measure("Parallel Matrix Multiplication", () -> {
            try {
                parallelMatrixMultiply(A, B);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }, NUM_THREADS);
    }

    @Benchmark
    public void benchmarkVectorizedMatrixMult() {
        vectorizedResult = measure("Vectorized Matrix Multiplication", () -> vectorizedMatrixMultiply(A, B), NUM_THREADS);
    }

    private Result measure(String algorithmName, Runnable matrixMultiplicationAlgorithm, int threadCount) {
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException e) { e.printStackTrace(); }

        OperatingSystemMXBean osBean = (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
        long beforeMemory = osBean.getCommittedVirtualMemorySize();

        double cpuBefore = osBean.getProcessCpuLoad() * 100;

        long startTime = System.nanoTime();
        matrixMultiplicationAlgorithm.run();
        long endTime = System.nanoTime();

        long afterMemory = osBean.getCommittedVirtualMemorySize();
        long memoryUsed = afterMemory - beforeMemory;

        double executionTime = (endTime - startTime) / 1e6;
        double cpuAfter = osBean.getProcessCpuLoad() * 100;

        return new Result(executionTime, memoryUsed, cpuBefore, cpuAfter, threadCount);
    }


    @TearDown(Level.Trial)
    public void tearDown() {
        if (basicResult != null) {
            printResults(size, "Basic Matrix Multiplication", basicResult);
        }

        if (parallelResult != null) {
            printResults(size, "Parallel Matrix Multiplication", parallelResult);
        }

        if (vectorizedResult != null) {
            printResults(size, "Vectorized Matrix Multiplication", vectorizedResult);
        }
    }

    private static void printResults(int size, String algorithmName, Result result) {
        System.out.println("\nAlgorithm: " + algorithmName);
        System.out.println("Matrix size: " + size + "x" + size);
        System.out.println("Execution time: " + result.executionTime + " ms");
        System.out.println("Threads used: " + result.threadCount);
        System.out.println("Memory used: " + String.format("%.2f", Math.abs(result.memoryUsed) / (1024.0 * 1024)) + " MB");
        System.out.println("CPU usage before: " + String.format("%.2f", result.cpuBefore) + " %");
        System.out.println("CPU usage after: " + String.format("%.2f", result.cpuAfter) + " %\n");
    }

    private static class Result {
        double executionTime;
        long memoryUsed;
        double cpuBefore;
        double cpuAfter;
        int threadCount;

        Result(double executionTime, long memoryUsed, double cpuBefore, double cpuAfter, int threadCount) {
            this.executionTime = executionTime;
            this.memoryUsed = memoryUsed;
            this.cpuBefore = cpuBefore;
            this.cpuAfter = cpuAfter;
            this.threadCount = threadCount;
        }
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(TestMatrixMultiplication.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
