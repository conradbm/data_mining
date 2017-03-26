import java.util.ArrayList;


public class ANN {

	int inputLayerCount;
	int endLayerCount;
	int numberHiddenLayers;
	int hiddenLayerCount;
	
	public ArrayList<Neuron> inputLayer;
	public ArrayList<ArrayList<Neuron>> hiddenLayer;
	public ArrayList<Neuron> outputLayer;
	
	
	//https://www.desmos.com/calculator
	public double S(double x, double t){
		return (1 / (1 + Math.pow(5, (-x+t))));
	}
	
	// Evaluate the neural net
	public ArrayList<Double> eval(ArrayList<Double> sampleInput){
		
		// add someInput as a 1-1 to the input layer inputBin
		
		// get sigmoid(startInput + thresh) for each neuron in input layer
		
		// add 2*weight for each neuron and pass it into the next layer
		return new ArrayList<Double>();
		
	}
	
	
	public ArrayList<Double> getNWeights(int n){
		// get N random floats and return an array list of doubles
		ArrayList<Double> d = new ArrayList<Double>();
		for(int i = 0; i < n; i++){
			d.add(Math.random()*2 - 1);
		}
		return d;
	}
	
	public void build(int inputLayerCount,
					  int endLayerCount,
					  int numberHiddenLayers,
					  int hiddenLayerCount){
		
		System.out.println("Constructing the ANN:\n");
		
		// 
		this.inputLayerCount = inputLayerCount;
		this.endLayerCount = endLayerCount;
		this.numberHiddenLayers = numberHiddenLayers;
		this.hiddenLayerCount = hiddenLayerCount;
		
		//
		this.inputLayer  = new ArrayList<Neuron>();
		this.hiddenLayer = new ArrayList<ArrayList<Neuron>>();
		this.outputLayer = new ArrayList<Neuron>();
		
		System.out.println("Adding neurons and layers.");
		
		// input layer
		System.out.println("Input Layer:");
		for(int i = 0; i < inputLayerCount; i++){
			System.out.println("Neuron_"+i+ " added to input layer.");
			Neuron n = new Neuron();
			n.threshold = Math.random()*2 - 1;
			this.inputLayer.add(n);
		}
		
		// hidden layer
		System.out.println("Hidden Layer:");
		for(int i = 0; i < numberHiddenLayers; i++){
			ArrayList<Neuron> layer_i = new ArrayList<Neuron>();
			System.out.println("Layer_"+i+ " added to hidden layer.");
			for(int j = 0; j < hiddenLayerCount; j++){
				System.out.println("Neuron_"+j+ " added to " + "Layer_" +i);
				Neuron n = new Neuron();
				n.threshold = Math.random()*2 - 1;
				layer_i.add(n);
			}
			this.hiddenLayer.add(layer_i);
		}
		
		// output layer
		System.out.println("Output Layer:");
		for(int i = 0; i < endLayerCount; i++){
			System.out.println("Neuron_"+i+ " added to the end layer.");
			Neuron n = new Neuron();
			n.threshold = Math.random()*2 - 1;
			this.outputLayer.add(n);
		}
		
		System.out.println("Finished adding neurons and layers.\n");
		
		System.out.println("Constructing weights and connections.");
		
		//input layer -> first hidden layer
		System.out.println("Connecting Input Layer to Hidden Layer 0");
		for(int i = 0; i < this.inputLayer.size(); i++){
			System.out.println("Connected Neuron_"+i +" to everything else.");
			this.inputLayer.get(i).outputNeurons.addAll(this.hiddenLayer.get(0));
			this.inputLayer.get(i).outputWeights = getNWeights(this.hiddenLayer.get(0).size());
		}
		
		//first hidden layer -> last hidden layer-1 || 0 -> size - 2
		
		for(int i = 0; i < this.hiddenLayer.size()-1; i++){
			System.out.println("Connecting Hidden Layer_"+i+" to Hidden Layer_"+(i+1));
			for(int j = 0; j < this.hiddenLayer.get(i).size(); j++){
				System.out.println("Connected Neuron_"+j +" to everything else.");
				this.hiddenLayer.get(i).get(j).outputNeurons.addAll(this.hiddenLayer.get(i+1));
				this.hiddenLayer.get(i).get(j).outputWeights = getNWeights(this.hiddenLayer.get(i+1).size());

			}
		}
		
		//hidden layer -> output layer
		System.out.println("Connecting Hidden Layer_" + (this.numberHiddenLayers-1) + " to the Output Layer.");
		for(int i = 0; i < this.hiddenLayer.get(this.numberHiddenLayers-1).size(); i++){
			System.out.println("Connected Neuron_"+i +" to everything else.");
			this.hiddenLayer.get(this.numberHiddenLayers-1).get(i).outputNeurons.addAll(this.outputLayer);
			this.hiddenLayer.get(this.numberHiddenLayers-1).get(i).outputWeights = getNWeights(this.outputLayer.size());
		}
		System.out.println("Done Constructing weights and connections.\n");
		System.out.println("Finished Constructing the ANN.");
	}
	
	public void printState(){
		
		System.out.println("\nPrinting the state of the ANN:\n");
		
		System.out.println("Input Layer");
		for(int i = 0; i < this.inputLayer.size(); i++){
			System.out.println("Neuron"+i);
			System.out.println("InputBin: " + this.inputLayer.get(i).inputBin);
			System.out.println("Weights: " + this.inputLayer.get(i).outputWeights);
			System.out.println("Threshold: "+ this.inputLayer.get(i).threshold);
		}
		
		System.out.println("Hidden Layer");
		for(int i = 0; i < this.numberHiddenLayers; i++){
			System.out.println("Layer"+i);
			for(int j = 0; j < this.hiddenLayer.get(i).size(); j++){
				System.out.println("Neuron"+j);
				System.out.println("InputBin: " + this.hiddenLayer.get(i).get(j).inputBin);
				System.out.println("Weights: "+this.hiddenLayer.get(i).get(j).outputWeights);
				System.out.println("Threshold: " + this.hiddenLayer.get(i).get(j).threshold);
			}
		}
		System.out.println("Output Layer");
		for(int i = 0; i < this.outputLayer.size(); i++){
			System.out.println("Neuron"+i);
			System.out.println("InputBin: " + this.outputLayer.get(i).inputBin);
			System.out.println("Weights: " + this.outputLayer.get(i).outputWeights);
			System.out.println("Threshold: "+ this.outputLayer.get(i).threshold);
		}
	}
}
