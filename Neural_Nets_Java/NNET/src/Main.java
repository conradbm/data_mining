import java.io.Serializable;


public class Main implements Serializable{

	
	
	public static void main(String[] args) {
		ANN nnet = new ANN();
		nnet.build(2, 1, 1, 3);
		nnet.printState();
	}
}
