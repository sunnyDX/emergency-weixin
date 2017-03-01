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

import com.xes.ml.LiblinearClassifier;
import com.xes.ml.LiblinearTrainer;
import com.xes.util.PartsOfSpeechTag;
import com.xes.util.RemoveWords;
import com.xes.util.SV2FeatureNodes;
import com.xes.util.Strings2StringArray;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
public class TextClassificationBasedOnLiblinear {
	/**
	 * 训练数据路径
	 */
	private static String trainDataPath = "xes-data/";
	/**
	 * 模型文件
	 */
	private static String liblinearModelFile = "ml-models/modelLiblinear";

	
	public static void main(String[] args) throws Exception {
		//去噪
		Pipe removepp=new RemoveWords();
		//分词
		Pipe segpp=new CNPipe(new CWSTagger("models/seg.m"));
		//String[]转化为Array[String]s	
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
		SeriesPipes pp = new SeriesPipes(new Pipe[]{removepp,segpp,s2spp,sparsepp,targetpp});	
		/**
		 * Liblinear
		 */
		System.out.print("\nLiblinear\n");
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
			
		System.out.print("Training Liblinear...\n");
		time_mark=System.currentTimeMillis();	
         
         // 模型参数
        SolverType solver = SolverType.L2R_LR_DUAL; // -s 0
        double C = 1.0;    // cost of constraints violation
        double eps = 0.001; // 停止标准
        Parameter parameter = new Parameter(solver, C, eps);
        LiblinearTrainer trainer = new LiblinearTrainer();
		//特征选择
		ItemFrequency tf = new ItemFrequency(trainset);
		FeatureSelect fs = new FeatureSelect(tf.getFeatureSize());
		fs.fS_CS(tf, 0.8f);
//		fs.fS_CS_Max(tf, 0.6f);
//		fs.fS_IG(tf, 0.8f);
		trainer.setFs(fs);
        LiblinearClassifier liblinear = (LiblinearClassifier)trainer.train(trainset, parameter);

        liblinear.saveTo(liblinearModelFile);
        liblinear = null;
        //load model or use it directly
     	System.out.print("Loading model...\n");
     	liblinear = LiblinearClassifier.loadFrom(liblinearModelFile);		
     	int count=0;
     	 
     	//预测并评测
 		for(int i=0;i<testset.size();i++){	
			Instance data = testset.getInstance(i);
			Integer gold = (Integer) data.getTarget();
			Predict<String> pres=liblinear.classify(data, Type.STRING, 1);
			String pred_label=pres.getLabel();
			String gold_label = liblinear.getLabel(gold); 			     	
			if(pred_label.equals(gold_label)){
				count++;
			}
			else{
				System.err.println(gold_label+"->"+pred_label+" : "+testset.getInstance(i).getSource().toString());
			}
		}
		System.out.println("..Testing Liblinear Complete");
		System.out.println("Liblinear Precision:"+((float)count/testset.size())+"("+count+"/"+testset.size()+")");
     	 
	}
}