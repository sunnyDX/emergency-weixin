package com.xes.util;

import java.util.Arrays;

import org.fnlp.ml.feature.FeatureSelect;
import org.fnlp.ml.types.Instance;
import org.fnlp.ml.types.InstanceSet;
import org.fnlp.ml.types.sv.HashSparseVector;
import de.bwaldvogel.liblinear.FeatureNode;
import gnu.trove.map.hash.TIntFloatHashMap;

public class SV2FeatureNodes{
	public SV2FeatureNodes(){
	}
	
	public static InstanceSet trans2FeatureNodes(InstanceSet insts) {
		InstanceSet newInsts = new InstanceSet();
		for(Instance inst: insts){
			HashSparseVector sv =  (HashSparseVector) inst.getData();
			TIntFloatHashMap tHash = sv.data;
			int[] tKeys = tHash.keys();
			Arrays.sort(tKeys);
			FeatureNode[] fns = new FeatureNode[tKeys.length];
	        for(int j=0;j<tKeys.length;j++) {  	
	            fns[j] = new FeatureNode(tKeys[j]+1,tHash.get(tKeys[j]));//特征索引Index是1开始计数
	        }
			inst.setData(fns);
			newInsts.add(inst);
		}
		return newInsts;
	}
	
	public static InstanceSet trans2FeatureNodes(InstanceSet insts,FeatureSelect fs) {
		InstanceSet newInsts = new InstanceSet();
		for(Instance inst: insts){
			HashSparseVector sv =  (HashSparseVector) inst.getData();
	        sv = fs.select(sv);		
			TIntFloatHashMap tHash = sv.data;
			int[] tKeys = tHash.keys();
			Arrays.sort(tKeys);
			FeatureNode[] fns = new FeatureNode[tKeys.length];
	        for(int j=0;j<tKeys.length;j++) {  	
	            fns[j] = new FeatureNode(tKeys[j]+1,tHash.get(tKeys[j]));//特征索引Index是1开始计数
	        }
			inst.setData(fns);
			newInsts.add(inst);
		}
		return newInsts;
	}
	
	public static Instance trans2FeatureNodes(Instance inst) {
		HashSparseVector sv =  (HashSparseVector) inst.getData();
		TIntFloatHashMap tHash = sv.data;
		int[] tKeys = tHash.keys();
		Arrays.sort(tKeys);
		FeatureNode[] fns = new FeatureNode[tKeys.length];
        for(int j=0;j<tKeys.length;j++) {  	
            fns[j] = new FeatureNode(tKeys[j]+1,tHash.get(tKeys[j]));//特征索引Index是1开始计数
        }
		inst.setData(fns);
	    return inst;
	}
	
	public static Instance trans2FeatureNodes(Instance inst,FeatureSelect fs) {

		HashSparseVector sv =  (HashSparseVector) inst.getData();
        sv = fs.select(sv);		
		TIntFloatHashMap tHash = sv.data;
		int[] tKeys = tHash.keys();
		Arrays.sort(tKeys);
		FeatureNode[] fns = new FeatureNode[tKeys.length];
        for(int j=0;j<tKeys.length;j++) {  	
            fns[j] = new FeatureNode(tKeys[j]+1,tHash.get(tKeys[j]));//特征索引Index是1开始计数
        }
		inst.setData(fns);
		return inst;
	}
}
