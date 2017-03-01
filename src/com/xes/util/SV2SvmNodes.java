package com.xes.util;

import java.util.Arrays;

import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.sv.HashSparseVector;
import gnu.trove.map.hash.TIntFloatHashMap;
import libsvm.svm_node;

public class SV2SvmNodes{
	private static final long serialVersionUID = -792623916821968204L;
	public SV2SvmNodes(){
	}
	public static InstanceSet trans2SvmNodes(InstanceSet insts) {
		InstanceSet newInsts = new InstanceSet();
		for(Instance inst: insts){
			HashSparseVector sv =  (HashSparseVector) inst.getData();
			TIntFloatHashMap tHash = sv.data;
			int[] tKeys = tHash.keys();
			Arrays.sort(tKeys);
			svm_node[] svmNodes = new svm_node[tKeys.length];
	        for(int j=0;j<tKeys.length;j++) {  	
	        	svm_node node = new svm_node();
	        	node.index = tKeys[j];
	        	node.value = tHash.get(tKeys[j]);
	        	svmNodes[j] = node;
	        }
			inst.setData(svmNodes);
			newInsts.add(inst);
		}
		return newInsts;
	}
	
	public static InstanceSet trans2SvmNodes(InstanceSet insts,FeatureSelect fs) {
		InstanceSet newInsts = new InstanceSet();
		for(Instance inst: insts){
			HashSparseVector sv =  (HashSparseVector) inst.getData();
	        sv = fs.select(sv);		
			TIntFloatHashMap tHash = sv.data;
			int[] tKeys = tHash.keys();
			Arrays.sort(tKeys);
			svm_node[] svmNodes = new svm_node[tKeys.length];
	        for(int j=0;j<tKeys.length;j++) {  	
	        	svm_node node = new svm_node();
	        	node.index = tKeys[j];
	        	node.value = tHash.get(tKeys[j]);
	        	svmNodes[j] = node;
	        }
			inst.setData(svmNodes);
			newInsts.add(inst);
		}
		return newInsts;
	}
	
	public static Instance trans2SvmNodes(Instance inst) {
		HashSparseVector sv =  (HashSparseVector) inst.getData();
		TIntFloatHashMap tHash = sv.data;
		int[] tKeys = tHash.keys();
		Arrays.sort(tKeys);
		svm_node[] svmNodes = new svm_node[tKeys.length];
        for(int j=0;j<tKeys.length;j++) {  	
        	svm_node node = new svm_node();
        	node.index = tKeys[j];
        	node.value = tHash.get(tKeys[j]);
        	svmNodes[j] = node;
        }
		inst.setData(svmNodes);
	    return inst;
	}
	
	public static Instance trans2SvmNodes(Instance inst,FeatureSelect fs) {

		HashSparseVector sv =  (HashSparseVector) inst.getData();
        sv = fs.select(sv);		
		TIntFloatHashMap tHash = sv.data;
		int[] tKeys = tHash.keys();
		Arrays.sort(tKeys);
		svm_node[] svmNodes = new svm_node[tKeys.length];
        for(int j=0;j<tKeys.length;j++) {  	
        	svm_node node = new svm_node();
        	node.index = tKeys[j];
        	node.value = tHash.get(tKeys[j]);
        	svmNodes[j] = node;
        }
		inst.setData(svmNodes);
		return inst;
	}

	
}


