package com.xes.rt;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
/**
 * 消息定向指派主驱动类
 * @author dx
 */
public class Assignment {
	
	public static void main(String[] args) throws Exception {	
		BlockingQueue<String> queue = new LinkedBlockingQueue<String>();		
		/* 可以根据实际设置生产者的线程数
		 * 默认是两个生产者线程，负责不断从标准输入流中读入数据，并发送给阻塞队列
		 * 默认是两个消费者线程，负责不断从阻塞队列读入数据，并做模型测试。
		 */
		MessageProducer producer1 = new MessageProducer(queue);
		MessageProducer producer2 = new MessageProducer(queue);
		MessageConsumer consumer1;
		MessageConsumer consumer2;
		if(args.length == 1){//是否制定模型文件路径，默认是jar包同层目录
			consumer1 = new MessageConsumer(queue,args[0]);
			consumer2 = new MessageConsumer(queue,args[0]);
		}else{
			consumer1 = new MessageConsumer(queue);
			consumer2 = new MessageConsumer(queue);
		}
		// 借助Executors启动进程
		ExecutorService service = Executors.newCachedThreadPool();
		service.execute(producer1);
		service.execute(producer2);
		service.execute(consumer1);
		service.execute(consumer2);
		// 退出Executor
		service.shutdown();
	}
		 

}
