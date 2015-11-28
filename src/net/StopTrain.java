package net;

import java.util.Objects;
import java.util.Scanner;

public class StopTrain implements Runnable{
    private boolean end;

    public StopTrain(){
        end = false;
    }

    @Override
    public void run() {
        Scanner sc = new Scanner(System.in);
        boolean repeat = true;
        while (repeat){
            byte sml = sc.nextByte();
            System.out.println("pressed");
            if (Objects.equals(sml, ' ')){
                repeat = false;
                end = true;
            }
        }
    }

    public boolean isEnd(){
        return end;
    }
}
