package com.xes.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;

import org.fnlp.ml.types.Instance;
import org.fnlp.nlp.cn.tag.CWSTagger;
import org.fnlp.nlp.cn.tag.POSTagger;
import org.fnlp.nlp.pipe.Pipe;
import org.fnlp.util.exception.LoadModelException;

public class PartsOfSpeechTag extends Pipe {
	public POSTagger tag;
	public CWSTagger cws;
	
	public  PartsOfSpeechTag() throws LoadModelException {
		CWSTagger cws = new CWSTagger("models/seg.m");
		tag = new POSTagger(cws,"models/pos.m");
	}
	
	@Override
	public void addThruPipe(Instance inst) throws Exception {
		// TODO Auto-generated method stub
		ArrayList<String> newdata=new ArrayList<String>();
		String data =  (String) inst.getData();
		String[][] sa = tag.tag2Array(data);
		for(int j = 0; j<sa[0].length; j++){
			if(sa[1][j].equals("名词") || sa[1][j].equals("实体名") ){
				newdata.add(sa[0][j]);
			}
		}
		inst.setData(newdata);
	}

}
