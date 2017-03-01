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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.fnlp.ml.classifier.AbstractClassifier;
import org.fnlp.ml.classifier.LabelParser;
import org.fnlp.ml.classifier.Predict;
import org.fnlp.ml.classifier.LabelParser.Type;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.nlp.pipe.Pipe;
import com.xes.util.SV2SvmNodes;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

public class LibsvmClassifier extends AbstractClassifier implements Serializable {
	private static final long serialVersionUID = -6932505634565877158L;
	protected Pipe pipe;
	protected svm_model model;
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

	@Override
	public Predict classify(Instance instance, int i) {
		// TODO Auto-generated method stub	
    	Instance ins = null;
    	double[] score = new double[svm.svm_get_nr_class(model)];
    	int[] labels = new int[svm.svm_get_nr_class(model)];

    	if(fs != null)
    		ins = SV2SvmNodes.trans2SvmNodes(instance,fs);
    	else
    		ins = SV2SvmNodes.trans2SvmNodes(instance);
        int pre = (int)svm.svm_predict_probability(model, (svm_node[])ins.getData(), score);
        svm.svm_get_labels(model, labels);  
        int index = searchIndex(labels,pre);
     	//默认无论i设置多少，都返回一个预测结果
        Predict res = new Predict(1);
        res.add(Integer.valueOf(pre), (float) score[index]);     
		return res;
	}

	@Override
	public Predict classify(Instance instance, Type type, int i) {
		// TODO Auto-generated method stub
        Predict res = classify(instance, i);
        return LabelParser.parse(res, factory.DefaultLabelAlphabet(), type);
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
	 * 得到类标签
	 * @param idx 类标签对应的索引
	 * @return
	 */
	public String getLabel(int idx) {
		return factory.DefaultLabelAlphabet().lookupString(idx);
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

	public svm_model getModel() {
		return model;
	}

	public void setModel(svm_model svmModel) {
		this.model = svmModel;
	}
    public static LibsvmClassifier loadFrom(String file) throws Exception
    {
    	LibsvmClassifier cl = null;
            try
            {
                ObjectInputStream in = new ObjectInputStream(new BufferedInputStream(new FileInputStream(file)));
                cl = (LibsvmClassifier)in.readObject();
                in.close();
            }
            catch(Exception e)
            {
                throw new Exception(e);
            }
            return cl;
    }
}
