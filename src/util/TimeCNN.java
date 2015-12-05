package util;


public class TimeCNN{
    private long timeAll;
    private long timeLast;
    private boolean isCheck;

    public TimeCNN(){
        timeAll = 0;
        timeLast = 0;
    }

    public void start(){
        timeLast = System.currentTimeMillis();
        isCheck = false;
    }

    public long getTimeLast(){
        timeLast = System.currentTimeMillis() - timeLast;
        timeAll += timeLast;
        isCheck = true;
        return timeLast;
    }

    public long getTimeAll(){
        if (!isCheck){
            timeLast = System.currentTimeMillis() - timeLast;
            timeAll += timeLast;
        }
        return timeAll;
    }
}