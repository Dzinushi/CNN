package net;

import java.util.Scanner;

public class StopTrain implements Runnable{
    private volatile boolean end;

    public StopTrain(){
        end = false;
    }

    @Override
    public void run() {
        Scanner sc = new Scanner(System.in);
        while (!isEnd())
        {
            String result = sc.nextLine();
            if (result.equals("")){
                end = true;
            }
        }
    }

    public boolean isEnd(){
        return end;
    }

    public void setEnd() {
        end = true;
    }
}