package com.xes.pipe;

import java.io.File;
import org.fnlp.data.reader.FileReader;
import org.fnlp.data.reader.Reader;
import org.fnlp.ml.classifier.Predict;
import org.fnlp.ml.classifier.LabelParser.Type;
import org.fnlp.ml.classifier.bayes.ItemFrequency;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.nlp.cn.tag.CWSTagger;
import org.fnlp.nlp.pipe.NGram;
import org.fnlp.nlp.pipe.Pipe;
import org.fnlp.nlp.pipe.SeriesPipes;
import org.fnlp.nlp.pipe.StringArray2SV;
import org.fnlp.nlp.pipe.Target2Label;
import org.fnlp.nlp.pipe.nlp.CNPipe;
import com.xes.ml.LibsvmClassifier;
import com.xes.ml.LibsvmTrainer;
import com.xes.util.PartsOfSpeechTag;
import com.xes.util.RemoveWords;
import com.xes.util.Strings2StringArray;

import libsvm.svm_parameter;

public class TextClassificationBasedOnLibsvm {
	/**
	 * 训练数据路径
	 */
	private static String trainDataPath = "xes-data/";
	/**
	 * 模型文件
	 */
	private static String svmModelFile = "ml-models/modelLibsvm";

	
	public static void main(String[] args) throws Exception {
		//去噪
		Pipe removepp=new RemoveWords();
		//分词
		Pipe segpp=new CNPipe(new CWSTagger("models/seg.m"));
		//String[]转化为Array[String]
		Pipe s2spp=new Strings2StringArray();
		//词性标注
//		Pipe pst = new PartsOfSpeechTag();
		//建立字典管理器
		AlphabetFactory af = AlphabetFactory.buildFactory();
		//使用n元特征
//		Pipe ngrampp = new NGram(new int[] {2,3});
		//将字符特征转换成字典索引;	
		Pipe sparsepp=new StringArray2SV(af);
		//将目标值对应的索引号作为类别
		Pipe targetpp = new Target2Label(af.DefaultLabelAlphabet());	
		//建立pipe组合
		SeriesPipes pp = new SeriesPipes(new Pipe[]{removepp,segpp,s2spp,targetpp,sparsepp});
		
		/**
		 * Libsvm
		 */
		System.out.print("\nLibsvm\n");
		System.out.print("\nReading data......\n");
		long time_mark=System.currentTimeMillis();
		InstanceSet instset = new InstanceSet(pp,af);
			
		//读取数据集
		Reader reader = new FileReader(trainDataPath,"UTF-8",".data");
        //按照装载管道生成模型数据集
		instset.loadThruStagePipes(reader);
		
		System.out.print("..Reading data complete "+(System.currentTimeMillis()-time_mark)+"(ms)\n");
		
		//将数据集分为训练是和测试集
		System.out.print("Sspliting....");
		float percent = 0.9f;
		InstanceSet[] splitsets = instset.split(percent);
		
		InstanceSet trainset = splitsets[0];
		InstanceSet testset = splitsets[1];	
		System.out.print("..Spliting complete!\n");
			
		System.out.print("Training Svm...\n");
		time_mark=System.currentTimeMillis();

        //Libsvm 参数配置
	    svm_parameter param = new svm_parameter();	    
	    param.svm_type = svm_parameter.C_SVC;
	    param.kernel_type = svm_parameter.LINEAR;
	    param.cache_size = 100;
	    param.eps = 0.00001;
	    param.C = 1;
	    param.probability = 1;
	    
	    LibsvmTrainer trainer = new LibsvmTrainer();
	    
		//特征选择
		ItemFrequency tf = new ItemFrequency(trainset);
		FeatureSelect fs = new FeatureSelect(tf.getFeatureSize());
		fs.fS_CS(tf, 0.8f);
//		fs.fS_CS_Max(tf, 0.8f);
//		fs.fS_IG(tf, 0.8f);
		trainer.setFs(fs);
		//模型训练
		LibsvmClassifier svmModel = (LibsvmClassifier) trainer.train(trainset, param);	
		svmModel.saveTo(svmModelFile);
        //load model or use it directly
        svmModel = null;
        svmModel = LibsvmClassifier.loadFrom(svmModelFile);
        
     	int count=0;
 		for(int i=0;i<testset.size();i++){	
			Instance data = testset.getInstance(i);
			Integer gold = (Integer) data.getTarget();
			Predict<String> pres=svmModel.classify(data, Type.STRING, 1);
			String pred_label=pres.getLabel();
			String gold_label = svmModel.getLabel(gold); 		     	
			if(pred_label.equals(gold_label)){
				count++;	
			}
			else{
				System.err.println(gold_label+"->"+pred_label+" : "+testset.getInstance(i).getSource().toString());
			}
		}
		System.out.println("..Testing Libsvm Complete");
		System.out.println("Libsvm Precision:"+((float)count/testset.size())+"("+count+"/"+testset.size()+")");
	}
}