package util;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public abstract class ThreadTaskManager {
    private static int cpu = Runtime.getRuntime().availableProcessors();
    private static ExecutorService exService = Executors.newFixedThreadPool(cpu);
    private static int length;

    public abstract void start(int start, int end);

    public ThreadTaskManager(int length){
        ThreadTaskManager.length = length;
        threadCommand(cpu);
    }

    private static void run(Runnable command){
        exService.execute(command);
    }

    public static void stop(){
        exService.shutdown();
    }

    private void threadCommand(int cpu){
        final int[][] steps = calcCpuLength(cpu);

        CountDownLatch count = new CountDownLatch(steps.length);
        for (int i = 0; i < steps.length; i++) {
            final int cpuIndex = i;
            final int start = steps[cpuIndex][0];
            final int end = steps[cpuIndex][1];
            Runnable command = () -> {
                start(start, end);
                count.countDown();
            };
            run(command);
        }
        try {
            count.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    private static int[][] calcCpuLength(int cpu){
        int[][] steps = null;

        // число итераций меньше числа процессоров
        if (length < cpu){
            steps = new int[length][2];
            for (int i = 0; i < length; i++) {
                steps[i][0] = i;
                steps[i][1] = i+1;
            }
        }
        // число итераций кратно числу процессоров
        else if (length % cpu == 0){
            steps = new int[cpu][2];
            int step = length / cpu;
            for (int i = 0; i < cpu; i++) {
                steps[i][0] = i * step;
                steps[i][1] = (i+1) * step;
            }
        }
        // универсальный вариант
        else {
            int copyCpu = cpu - 1;

            boolean repeat = true;
            while(repeat){
                int step = length / copyCpu;
                int balance = length - (step * copyCpu);
                if (balance < step){
                    steps = new int[balance == 0 ? (copyCpu) : (copyCpu + 1)][2];
                    for (int j = 0; j < copyCpu; j++) {
                        steps[j][0] = j * step;
                        steps[j][1] = (j + 1) * step;
                    }
                    if (balance != 0){
                        steps[copyCpu][0] = length - balance;
                        steps[copyCpu][1] = length;
                    }
                    repeat = false;
                }
                else {
                    --copyCpu;
                }
            }
        }

        return steps;
    }
}