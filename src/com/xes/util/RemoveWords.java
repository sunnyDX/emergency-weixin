package com.xes.util;

import org.fnlp.ml.types.Instance;
import org.fnlp.nlp.pipe.Pipe;

public class RemoveWords extends Pipe{

	private static final long serialVersionUID = -859010428210775557L;
	String[] list=new String[]{"技术"};
	public void addThruPipe(Instance inst) {
		String data =  (String) inst.getData();
		for(int i=0;i<list.length;i++){
			String str=list[i];
			data=data.replaceAll(str, "");
		}
		inst.setData(data);
	}
}
