import java.util.ArrayList;

public class Neuron {
	
	// Neuron connection to the next layer
	public ArrayList<Neuron> outputNeurons = new ArrayList<Neuron>();
	
	// Weights of each connection
	public ArrayList<Double> outputWeights = new ArrayList<Double>();
	
	// Sum of all the previous layers weights into this neuron
	double inputBin = 0;
	
	//
	double threshold = 0;
	
}
