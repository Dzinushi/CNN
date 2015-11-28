package util;

public class TimeCNN {
    private long timeAll;
    private long timeLast;

    public TimeCNN(){
        timeAll = 0;
        timeLast = 0;
    }

    public void start(){
        timeLast = System.currentTimeMillis();
    }

    public long getTimeLast(){
        timeLast = System.currentTimeMillis() - timeLast;
        timeAll += timeLast;
        return timeLast;
    }

    public long getTimeAll(){
        return timeAll;
    }
}
