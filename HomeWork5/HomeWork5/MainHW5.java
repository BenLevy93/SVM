package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

class MaxKernel {
	protected double TPR;
	protected double FPR;
	protected double power;
	protected double param;
	protected MainHW5.KernelType type;

	public MaxKernel(double TPR, double FPR, double power, double param, MainHW5.KernelType type) {
		this.TPR = TPR;
		this.FPR = FPR;
		this.power = power;
		this.param = param;
		this.type = type;
	}
}

public class MainHW5 {

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances data = loadData("cancer.txt");
		BufferedReader buff = readDataFile("cancer.txt");
		Random rand = new Random(0);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		Instances trainingData = randData.trainCV(5,1);
		Instances testingData  = randData.testCV(5,1);
		SVM svm = new SVM();
		double alpha = 1.5;
		double[] gammaVal = new double[]{0.005, 0.05, 0.5};
		double[] degVal = new double[]{2, 3, 4};

		MaxKernel poly = getBestKernel(svm, trainingData, testingData, degVal, alpha, KernelType.POLY);
		MaxKernel rbf = getBestKernel(svm, trainingData, testingData, gammaVal, alpha, KernelType.RBF);

		if (poly.power > rbf.power) {
			getMaxSlack(svm,trainingData,testingData,poly);
		} else {
			getMaxSlack(svm,trainingData,testingData,rbf);
		}


	}

	private static void getMaxSlack(SVM svm, Instances trainingData, Instances testingData, MaxKernel kernel) throws Exception {
		int[] error;
		double TP = 0;
		double FP = 0;
		if (kernel.type == KernelType.POLY) {
			PolyKernel ker = new PolyKernel();
			ker.setExponent(kernel.param);
			svm.setKernel(ker);
			System.out.println("The best kernel is: Poly kernel parameter " + kernel.param + " " + kernel.power);
		} else {
			RBFKernel ker = new RBFKernel();
			ker.setGamma(kernel.param);
			svm.setKernel(ker);
			System.out.println("The best kernel is: RBF kernel parameter " + kernel.param + " " + kernel.power);
		}
		for (int i = -4; i < 2; i++) {
			for (double j = 1; j < 4; j++) {
				double c = Math.pow(10, i) * (j / 3.0);
				svm.setC(c);
				svm.buildClassifier(trainingData);
				error = svm.calcConfusion(testingData);
				System.out.println("For C " + c + " the rates are");
				System.out.println("TPR : " + getTPR(error));
				System.out.println("FPR : " + getFPR(error));
				System.out.println();

			}
		}

	}


	private static MaxKernel getBestKernel(SVM svm, Instances trainingData, Instances testingData, double[] values, double alpha, KernelType type) throws Exception {
		int[] error;
		double TP = 0;
		double FP = 0;
		double maxParam = 0;
		double power = 0;
		String msg;
		PolyKernel kernel = new PolyKernel();
		RBFKernel kernelRBF = new RBFKernel();
		for (double param : values) {
			if (type == KernelType.POLY) {
				kernel.setExponent(param);
				svm.setKernel(kernel);
				msg = "For PolyKernel with exponent ";
			} else {
				kernelRBF.setGamma(param);
				svm.setKernel(kernelRBF);
				msg = "For RBFKernel with gamma ";

			}
			svm.buildClassifier(trainingData);
			error = svm.calcConfusion(testingData);
			double TPR = getTPR(error);
			double FPR = getFPR(error);

			System.out.println(msg + param + " the rates are:\n" +
					"TPR = " + TPR + "\n" +
					"FPR = " + FPR + "\n");
			double curPower = (alpha * TPR) - FPR;
			if (curPower > power) {
				TP = TPR;
				FP = FPR;
				power = curPower;
				maxParam = param;
			}
		}
		return new MaxKernel(TP, FP, power, maxParam, type);
	}


	private static double getFPR(int[] error) {
		double FP = error[1];
		double TN = error[2];
		return FP / (FP + TN);
	}

	private static double getTPR(int[] error) {
		double TP = error[0];
		double FN = error[3];
		return TP / (TP + FN);
	}

	public enum KernelType {
		POLY,
		RBF
	}
}
