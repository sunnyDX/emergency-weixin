package com.xes.ml;

import org.fnlp.ml.classifier.AbstractClassifier;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.nlp.pipe.Pipe;
import org.fnlp.nlp.pipe.SeriesPipes;

import com.xes.util.SV2SvmNodes;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class LibsvmTrainer {   
	
	protected FeatureSelect fs;

	public FeatureSelect getFs() {
		return fs;
	}

	public void setFs(FeatureSelect fs) {
		this.fs = fs;
	}
	public LibsvmTrainer()
	{
	}

	public AbstractClassifier train(InstanceSet trainset,svm_parameter parameter)
	{
	    AlphabetFactory af = trainset.getAlphabetFactory();
	    SeriesPipes pp = (SeriesPipes)trainset.getPipes();
	    pp.removeTargetPipe();
	    return train(trainset, af, ((Pipe) (pp)),parameter);
	}
	
	public AbstractClassifier train(InstanceSet trainset, AlphabetFactory af, Pipe pp,svm_parameter parameter)
	{
		InstanceSet newTrainset = null;
		//将sv格式的数据集换成svm_node数据集，以便Libsvm识别。
		if(fs != null)
			newTrainset=SV2SvmNodes.trans2SvmNodes(trainset,fs);
		else
			newTrainset=SV2SvmNodes.trans2SvmNodes(trainset);
		LibsvmClassifier classifier = new LibsvmClassifier();	    
		svm_problem problem = new svm_problem();
	    problem.l = newTrainset.size(); // 训练样本数
	    problem.x = new svm_node[problem.l][]; //特征二维数组
	    problem.y = new double[problem.l]; //标签一维数组	    
	    for(int i=0; i<problem.l; i++){
		        problem.y[i] = (Integer)trainset.get(i).getTarget();
		        problem.x[i] = (svm_node[])trainset.get(i).getData();   	
		}	    
	    System.out.println(svm.svm_check_parameter(problem, parameter));
	    svm_model svmModel = svm.svm_train(problem, parameter);
	    classifier.setFactory(af);
	    classifier.setPipe(pp);
	    classifier.setModel(svmModel);
	    classifier.setFs(fs);  
	    return classifier;
	}
}
