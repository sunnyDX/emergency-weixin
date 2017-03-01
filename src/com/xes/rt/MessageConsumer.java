package com.xes.rt;

import java.lang.reflect.Method;
import java.util.concurrent.BlockingQueue;

import org.fnlp.ml.classifier.AbstractClassifier;
import org.fnlp.ml.classifier.Predict;
import org.fnlp.ml.classifier.LabelParser.Type;
import org.fnlp.ml.types.Instance;
import org.fnlp.nlp.pipe.Pipe;

import com.xes.ml.LiblinearClassifier;
/**
 * 消息消费者进程
 * @author dengxing
 */
public class MessageConsumer  implements Runnable{
	//阻塞消息队列
	private BlockingQueue<String> queue;
	//分隔符
	private static final String SEPARATOR = "\001";
	//抽象分类器
	private AbstractClassifier classifier;
	//管道
	private Pipe p;
	
	public MessageConsumer(BlockingQueue<String> queue,String modelFile) throws Exception {
		this.queue= queue;
		init(modelFile);
	}
	public MessageConsumer(BlockingQueue<String> queue) throws Exception {
		this.queue= queue;
		String defaultFile = "ml-models/modelLiblinear";
		init(defaultFile);
	}
	
    /**
     * 通过反射获取分类器信息
     * 目的是能够直接根据模型文件参数获取相应的分类器信息和管道
     * @param path
     * @throws Exception
     */
	public void init(String path) throws Exception{
		String[] str = path.split("/");
		Class<?> c = null;
		if(str[str.length-1].contains("gz")){
			String modelName = str[str.length-1].replace(".gz","").substring(5);
			String className  = modelName+"Classifier";
			c = Class.forName("org.fnlp.ml.classifier."+modelName.toLowerCase()+"."+className);
		}else{
			String className  = str[str.length-1].substring(5)+"Classifier";
			c = Class.forName("com.xes.ml."+className);
		}
//		System.out.println(c.getName());
		Method method1 = c.getMethod("loadFrom", String.class);	
		Method method2 = c.getMethod("getPipe", null);	
		classifier = (AbstractClassifier)method1.invoke(null, new Object[] {path});			
		p = ((Pipe) method2.invoke(classifier,null));
	}
	
	/**
	 * Unix时间戳转化
	 * @param timestampString
	 * @return
	 */
	public String timeStamp2Date(String timestampString){  
		Long timestamp = Long.parseLong(timestampString);  
		String date = new java.text.SimpleDateFormat("dd/MM/yyyy HH:mm:ss").format(new java.util.Date(timestamp));  
		return date;  
	}
	
	
	/**
	 * 根据分类器做消息定向指派
	 * @param message
	 * @return
	 * @throws Exception
	 */
	public String getAssignment(String message) {
		Instance inst = new Instance(message);
		try {
			//特征转换
			p.addThruPipe(inst);
		} catch (Exception e) {
			e.printStackTrace();   
		}
		
		Predict<String> pre = (Predict<String>) classifier.classify(inst, Type.STRING, 1);
		String res = pre.getLabel();
		double score = pre.getScore(0);
		return res+SEPARATOR+score;
	}
	 
	public void run() {
		while(true){	    
			try{  
			    String message = queue.take();		    
			    if(message !=null){
			    	String[] strs = message.split(SEPARATOR);	    	
			    	String creatTime = timeStamp2Date(strs[2]);
			    	String chatRoom = strs[3].split("@")[0];
			    	String talker = strs[4].split(":\\n")[0];
			    	String context = strs[4].split(":\\n")[1];
//			    	System.out.println(context.contains("？"));
			    	if(context.contains("？")){
				    	if(context.contains("@")){
				    		String as = context.split(" ")[0].replace("@", "");
				    		System.out.println("聊天室ID："+chatRoom+SEPARATOR+
				    						   "发话时间："+creatTime+SEPARATOR+
				    				           "发言者："+talker+SEPARATOR+
				    				           "消息内容："+context+SEPARATOR+
				    				           "主动指派："+as+SEPARATOR+
				    				           "定向指派："+getAssignment(context));
				    	}else
				    		System.out.println("聊天室ID："+chatRoom+SEPARATOR+
		    						           "发话时间："+creatTime+SEPARATOR+
				    				           "发言者："+talker+SEPARATOR+
				    						   "消息内容："+context+SEPARATOR+
				    						   "主动指派：NULL"+SEPARATOR+
				    						   "定向指派："+getAssignment(context));	
			    	}

			    }
			}catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
		
}
