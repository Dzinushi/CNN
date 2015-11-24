package util;

public class Precision {
    int value;
    int count;

    public Precision(){
        value = 0;
    }

    public void increase() {
        ++this.value;
        ++this.count;
    }

    public void clear(){
        this.value = 0;
        this.count = 0;
    }

    public int getValue() {
        return value;
    }

    public int getCount() {
        return count;
    }
}
