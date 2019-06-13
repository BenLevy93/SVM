package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}

	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}

	public int[] calcConfusion(Instances instances) throws Exception{
		int [] error = new int[4];//[TP, FP, TN, FN]
		for(Instance instance : instances){
			int result = getConfusion(m_smo.classifyInstance(instance), instance.classValue());
			error[result]++;
		}

		return error;
	}

	private int getConfusion(double prediction, double realVal) {
		int result;
		if (prediction == 1.0) {
			if (realVal == 1.0) {
				result = 0;//TP
			} else {
				result = 1;//FP
			}
		} else {
			if (realVal == 0) {
				result = 2;//TN
			} else {
				result = 3;//FN
			}
		}
		return result;
	}

	public void setKernel(Kernel value){
		this.m_smo.setKernel(value);
	}

	public void setC(double c){
		this.m_smo.setC(c);
	}

	public double getC(){
		return this.m_smo.getC();
	}

}
