package util;

public class Precision {
    int value;
    int count;

    public Precision(){
        value = 0;
    }

    public void increase() {
        ++this.value;
    }

    public void setCount(int count){
        this.count = count;
    }

    public void resetValue(){
        this.value = 0;
    }

    public int getValue() {
        return value;
    }

    public int getCount() {
        return count;
    }
}
