package com.xes.ml;

import org.fnlp.ml.classifier.AbstractClassifier;
import org.fnlp.ml.classifier.bayes.ItemFrequency;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.nlp.pipe.Pipe;
import org.fnlp.nlp.pipe.SeriesPipes;

import com.xes.util.SV2FeatureNodes;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;

public class LiblinearTrainer {   
	
	protected FeatureSelect fs;

	public FeatureSelect getFs() {
		return fs;
	}

	public void setFs(FeatureSelect fs) {
		this.fs = fs;
	}
	public LiblinearTrainer()
	{
	}

	public AbstractClassifier train(InstanceSet trainset,Parameter parameter)
	{
	    AlphabetFactory af = trainset.getAlphabetFactory();
	    SeriesPipes pp = (SeriesPipes)trainset.getPipes();  
	    return train(trainset, af, pp, parameter);
	}
	
	public AbstractClassifier train(InstanceSet trainset, AlphabetFactory af, Pipe pp,Parameter parameter)
	{
		InstanceSet newTrainset = null;
		//将sv格式的数据集换成FeatureNodes数据集，以便Liblinear识别。
		if(fs != null)
			newTrainset=SV2FeatureNodes.trans2FeatureNodes(trainset,fs);
		else
			newTrainset=SV2FeatureNodes.trans2FeatureNodes(trainset);
	    LiblinearClassifier classifier = new LiblinearClassifier();	    
	    Problem problem = new Problem();
	    problem.l = newTrainset.size(); // 训练样本数
	    problem.n = af.getFeatureSize(); // 特征维数
	    problem.x = new FeatureNode[problem.l][]; //特征二维数组
	    problem.y = new int[problem.l]; //标签一维数组
	     for(int i=0; i<problem.l; i++){
	        problem.y[i] = (Integer)newTrainset.get(i).getTarget();
	        problem.x[i] = (FeatureNode[])newTrainset.get(i).getData();   	
	     }
	    Model lmodel = Linear.train(problem, parameter);
	    af.setStopIncrement(true);
	    classifier.setFactory(af);
	    classifier.setPipe(pp);
	    classifier.setModel(lmodel);
	    classifier.setFs(fs);
	    return classifier;
	}
}
