package com.xes.pipe;

import gnu.trove.iterator.TIterator;

import java.io.File;
import java.sql.Time;

import org.fnlp.data.reader.DocumentReader;
import org.fnlp.data.reader.FileReader;
import org.fnlp.data.reader.Reader;
import org.fnlp.data.reader.SimpleFileReader;
import org.fnlp.ml.classifier.Predict;
import org.fnlp.ml.classifier.LabelParser.Type;
import org.fnlp.ml.classifier.bayes.BayesClassifier;
import org.fnlp.ml.classifier.bayes.BayesTrainer;
import org.fnlp.ml.classifier.bayes.ItemFrequency;
import org.fnlp.ml.classifier.knn.KNN;
import org.fnlp.ml.classifier.knn.KNNClassifier;
import org.fnlp.ml.classifier.linear.Linear;
import org.fnlp.ml.classifier.linear.OnlineTrainer;
import org.fnlp.ml.eval.Evaluation;
import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.alphabet.AlphabetFactory;
import org.fnlp.ml.types.alphabet.IFeatureAlphabet;
import org.fnlp.ml.types.alphabet.StringFeatureAlphabet;
import org.fnlp.nlp.cn.tag.CWSTagger;
import org.fnlp.nlp.pipe.NGram;
import org.fnlp.nlp.pipe.Pipe;
import org.fnlp.nlp.pipe.SeriesPipes;
import org.fnlp.nlp.pipe.StringArray2IndexArray;
import org.fnlp.nlp.pipe.StringArray2SV;
import org.fnlp.nlp.pipe.Target2Label;
import org.fnlp.nlp.pipe.nlp.CNPipe;
import org.fnlp.nlp.similarity.SparseVectorSimilarity;

import com.xes.util.PartsOfSpeechTag;
import com.xes.util.RemoveWords;
import com.xes.util.Strings2StringArray;

public class TextClassificationBasedOnKNN {
	/**
	 * 训练数据路径
	 */

	private static String trainDataPath = "xes-data/";
	/**
	 * 模型文件
	 */
	private static String knnModelFile = "ml-models/modelKNN.gz";

	
	public static void main(String[] args) throws Exception {
		//去噪
		Pipe removepp=new RemoveWords();
		//分词
		Pipe segpp=new CNPipe(new CWSTagger("models/seg.m"));
		//String[]转化为Array[String]
		Pipe s2spp=new Strings2StringArray();
		//词性标注
		Pipe pst = new PartsOfSpeechTag();
		//建立字典管理器
		AlphabetFactory af = AlphabetFactory.buildFactory();
		//使用n元特征
		Pipe ngrampp = new NGram(new int[] {2,3});
		//将字符特征转换成字典索引;	
		Pipe sparsepp=new StringArray2SV(af);
		//将目标值对应的索引号作为类别
		Pipe targetpp = new Target2Label(af.DefaultLabelAlphabet());	
		//建立pipe组合
		SeriesPipes pp = new SeriesPipes(new Pipe[]{removepp,segpp,s2spp,targetpp,sparsepp});

		/**
		 * Knn
		 */
		System.out.print("\nKnn\n");
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
		float percent = 0.8f;
		InstanceSet[] splitsets = instset.split(percent);
		
		InstanceSet trainset = splitsets[0];
		InstanceSet testset = splitsets[1];	
		System.out.print("..Spliting complete!\n");
		
		
		System.out.print("Training Knn...\n");
		time_mark=System.currentTimeMillis();
		//余弦相似度计算方法
		SparseVectorSimilarity sim=new SparseVectorSimilarity();
		pp.removeTargetPipe();
		KNNClassifier knn=new KNNClassifier(trainset, pp, sim, af, 10);	
		//特征选择
		ItemFrequency tf=new ItemFrequency(trainset);
		FeatureSelect fs=new FeatureSelect(tf.getFeatureSize());
//		fs.fS_CS(tf, 0.8f);
//		fs.fS_CS_Max(tf, 0.8f);
		fs.fS_IG(tf, 0.8f);
		knn.setFs(fs);
		af.setStopIncrement(true);	
				
		long time_train=System.currentTimeMillis()-time_mark;
		
		System.out.print("..Training compelte!\n");
		System.out.print("Saving model...\n");
		
		knn.saveTo(knnModelFile);	
			
		knn = null;
		System.out.print("..Saving model compelte!\n");

		System.out.print("Loading model...\n");
		knn =KNNClassifier.loadFrom(knnModelFile);
			

		
		System.out.print("..Loading model compelte!\n");
		System.out.println("Testing Knn...\n");
		int count=0;

		for(int i=0;i<testset.size();i++){
			Instance data = testset.getInstance(i);
			Integer gold = (Integer) data.getTarget();
			Predict<String> pres=(Predict<String>) knn.classify(data, Type.STRING, 3);
			String pred_label=pres.getLabel();
			String gold_label = knn.getLabel(gold);
			
			if(pred_label.equals(gold_label)){
				count++;
			}
			else{
				System.err.println(gold_label+"->"+pred_label+" : "+testset.getInstance(i).getSource().toString());
			}
		}
		int knnCount=count;
		System.out.println("..Testing Knn Complete");
		System.out.println("Knn Precision:"+((float)knnCount/testset.size())+"("+knnCount+"/"+testset.size()+")");

	}

}
