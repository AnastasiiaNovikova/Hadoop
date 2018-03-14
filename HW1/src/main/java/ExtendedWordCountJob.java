import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.commons.codec.binary.Base64;

import java.io.IOException;
import java.util.HashSet;
import java.util.HashMap;
import java.util.regex.*;

public class ExtendedWordCountJob extends Configured implements Tool {
    @Override
    public int run(String[] args) throws Exception {
        Job job = GetJobConf(getConf(), args[0], args[1]);
        return job.waitForCompletion(true) ? 0 : 1;
    }

    static public class ExtendedWordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
		private int CACHE_SIZE = 100000000;
        private HashMap<String, IntWritable> cache = null;
		private HashSet<String> docCache = null;
		private Pattern pattern = Pattern.compile("\\p{L}+");
		
		@Override
		protected void setup(Context context)
		{
			cache = new HashMap<String, IntWritable>();
			docCache = new HashSet<String>();
		}
		
		@Override
        protected void map(LongWritable key, Text rawData, Context context)
                throws IOException, InterruptedException
        {
			Matcher matcher = pattern.matcher(rawData.toString());
			while (matcher.find()) {
				docCache.add(matcher.group(0).toLowerCase());
			}
			
			for (String word : docCache)
			{
				if (cache.containsKey(word))
				{
					IntWritable current = cache.get(word);
					current.set(current.get() + 1);
				}
				else
				{
					cache.put(word, new IntWritable(1));
				}
			}
			docCache.clear();
				
			if (cache.size() >= CACHE_SIZE)
			{
				dropHash(context);
			}
        }
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException
		{
			dropHash(context);
		}
		
		private void dropHash(Context context) throws IOException, InterruptedException
		{
			for (String term : cache.keySet())
			{
				IntWritable count = cache.get(term);
				context.write(new Text(term), count);
			}
			cache.clear();
		}
    }
	
	static public class ExtendedWordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException
        {
			int sum = 0;
			for (IntWritable value : values) {
				sum += value.get();
			}
            context.write(key, new IntWritable(sum));
        }
    }

    public static Job GetJobConf(Configuration conf, String input, String out_dir) throws IOException {
        Job job = Job.getInstance(conf);
        job.setJarByClass(ExtendedWordCountJob.class);
        job.setJobName(ExtendedWordCountJob.class.getCanonicalName());

        job.setInputFormatClass(SplittedGzipInputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(out_dir));

        job.setMapperClass(ExtendedWordCountMapper.class);
		job.setReducerClass(ExtendedWordCountReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        return job;
    }

    public static void main(String[] args) throws Exception {
        int exitCode = ToolRunner.run(new ExtendedWordCountJob(), args);
        System.exit(exitCode);
    }
}
