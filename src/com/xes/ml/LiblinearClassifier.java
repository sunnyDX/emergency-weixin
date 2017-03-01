package com.xes.ml;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;

import org.fnlp.ml.classifier.AbstractClassifier;
import org.fnlp.ml.classifier.LabelParser;
import org.fnlp.ml.classifier.Predict;
import org.fnlp.ml.classifier.LabelParser.Type;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.classifier.TPredict;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.nlp.pipe.Pipe;

import com.xes.util.SV2FeatureNodes;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import libsvm.svm;

public class LiblinearClassifier extends AbstractClassifier implements Serializable{
	private static final long serialVersionUID = 6383778878463082743L;
	protected Pipe pipe;
	protected Model model;
	protected FeatureSelect fs;

	public FeatureSelect getFs() {
		return fs;
	}

	public void setFs(FeatureSelect fs) {
		this.fs = fs;
	}

	public Pipe getPipe() {
		return pipe;
	}

	public void setPipe(Pipe pipe) {
		this.pipe = pipe;
	}

	
    public void setFactory(AlphabetFactory factory)
    {
        this.factory = factory;
    }

	public Predict classify(Instance instance, int i) {
		// TODO Auto-generated method stub	
    	Instance ins = null;
    	double[] score = new double[model.getNrClass()];
    	int[] labels = model.getLabels();

    	if(fs != null)
    		ins = SV2FeatureNodes.trans2FeatureNodes(instance,fs);
    	else
    		ins = SV2FeatureNodes.trans2FeatureNodes(instance);
        int pre =  Linear.predictProbability(model, (FeatureNode[])ins.getData(), score);
        int index = searchIndex(labels, pre);
        Predict res = new Predict(1);
        res.add(Integer.valueOf(pre), (float) score[index]);   
		return res;
	}

	public Predict classify(Instance instance, Type type, int i) {
		// TODO Auto-generated method stub
        Predict res = classify(instance, i);
        return LabelParser.parse(res, factory.DefaultLabelAlphabet(), type);
	}
	
	/**
	 * 得到类标签
	 * @param idx 类标签对应的索引
	 * @return
	 */
	public String getLabel(int idx) {
		return factory.DefaultLabelAlphabet().lookupString(idx);
	}

	public int searchIndex(int[] array,int value){
		for(int i =0;i<array.length;i++){
			if(array[i] == value){
				return i;
			}
		}
		return -1;
	}
	/**
	 * 将分类器保存到文件
	 * @param file
	 * @throws IOException
	 */
	public void saveTo(String file) throws IOException {
		File f = new File(file);
		File path = f.getParentFile();
		if(!path.exists()){
			path.mkdirs();
		}
		
		ObjectOutputStream out = new ObjectOutputStream(
				new BufferedOutputStream(new FileOutputStream(file)));
		out.writeObject(this);
		out.close();
	}

	public Model getModel() {
		return model;
	}

	public void setModel(Model model) {
		this.model = model;
	}
    public static LiblinearClassifier loadFrom(String file) throws Exception
    {
    	LiblinearClassifier cl = null;
        try
        {
            ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(file)));
            cl = (LiblinearClassifier)in.readObject();
            in.close();
        }
        catch(Exception e)
        {
            throw new Exception(e);
        }
        return cl;
    }

}
