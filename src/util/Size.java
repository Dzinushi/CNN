package util;

import java.io.Serializable;


public class Size implements Serializable{

    public int x;
    public int y;

    public Size(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public Size(Size size){
        if (size != null) {
            copy(size);
        }
    }

    public String toString() {
        return "[" + x  + ";" + y + "]";
    }

    public Size divide(Size scaleSize) {
        int x = this.x / scaleSize.x;
        int y = this.y / scaleSize.y;
        if (x * scaleSize.x != this.x || y * scaleSize.y != this.y)
            throw new RuntimeException(this + "Error " + scaleSize);
        return new Size(x, y);
    }

    public Size subtract(Size size, int append) {
        int x = this.x - size.x + append;
        int y = this.y - size.y + append;
        return new Size(x, y);
    }

    public void copy(Size mapsSize) {
        this.x = mapsSize.x;
        this.y = mapsSize.y;
    }
}