package util;


public class Size{

    public final int x;
    public final int y;

    public Size(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public String toString() {
        StringBuilder s = new StringBuilder("Size(").append(" x = ")
                .append(x).append(" y= ").append(y).append(")");
        return s.toString();
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
}
