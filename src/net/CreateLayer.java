package net;

import java.util.ArrayList;
import java.util.List;

public class CreateLayer {
    private List<Layer> layers;

    public CreateLayer() {
        layers = new ArrayList<>();
    }

    public CreateLayer createLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public List<Layer> getListLayers(){
        return layers;
    }
}
