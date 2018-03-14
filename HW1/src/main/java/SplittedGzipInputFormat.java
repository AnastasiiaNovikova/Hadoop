import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.EOFException;
import java.io.DataInput;
import java.io.DataOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.Inflater;

public class SplittedGzipInputFormat extends FileInputFormat<LongWritable, Text>
{
	static public class IndexedSplit extends FileSplit
	{
		private ArrayList<Integer> index;
		private int startDocId;
		private int endDocId;

        public IndexedSplit()
		{
			super();
		}

        public IndexedSplit(Path path, long offset, long length, ArrayList<Integer> index,
			int startDocId, int endDocId)
		{
            super(path, offset, length, new String[] {});
            this.index = index;
			this.startDocId = startDocId;
			this.endDocId = endDocId;
        }

        @Override
        public void write(DataOutput out) throws IOException
		{
            super.write(out);
            out.writeInt(index.size());
            for (int pageSz: index)
			{
                out.writeInt(pageSz);
            }
			out.writeInt(startDocId);
			out.writeInt(endDocId);
        }

        @Override
        public void readFields(DataInput in) throws IOException
		{
            super.readFields(in);
            int pagesCount = in.readInt();
            index = new ArrayList<Integer>();

            for (int page = 0; page < pagesCount; page++)
			{
                index.add(in.readInt());
            }
			startDocId = in.readInt();
			endDocId = in.readInt();
        }

        public ArrayList<Integer> getIndex()
		{
            return index;
        }
		
		public int getStartDocId()
		{
			return startDocId;
		}
		
		public int getEndDocId()
		{
			return endDocId;
		}
	}
	
	public class GzipRecordReader extends RecordReader<LongWritable, Text>
	{
		FSDataInputStream input;
		int curPos = 0;
		int endDocId = 0;
        long bytesRead = 0L;
		long totalSize = 0L;
		long positionInFile = 0L;
		ArrayList<Integer> index;
		LongWritable key = new LongWritable();
        Text value = new Text();
		byte[] compressionBuf = new byte[8096];
		
		int nextCounter = 0;

        @Override
        public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException
		{
            Configuration conf = context.getConfiguration();
            IndexedSplit fsplit = (IndexedSplit)split;
            Path path = fsplit.getPath();
            FileSystem fs = path.getFileSystem(conf);

            input = fs.open(path);
			positionInFile = fsplit.getStart();
            input.seek(positionInFile);
			index = fsplit.getIndex();
			totalSize = fsplit.getLength();
			curPos = fsplit.getStartDocId();
			endDocId = fsplit.getEndDocId();
			// System.out.println("Initialized RecordReader start index position = " + curPos + ", end position = " + endDocId + ", index size = " + index.size());
        }

        @Override
        public boolean nextKeyValue() throws IOException, InterruptedException
		{
            if (curPos <= endDocId)
			{
				int zippedDocumentLen = index.get(curPos);
				byte[] dataBuffer = new byte[zippedDocumentLen];
				IOUtils.readFully(input, dataBuffer, 0, zippedDocumentLen);
				bytesRead += zippedDocumentLen;
				key.set(positionInFile);
				curPos++;
				positionInFile += zippedDocumentLen;

				try
				{
					value.set(processCompressedDoc(dataBuffer));
				}
				catch (java.util.zip.DataFormatException e)
				{
					throw new IOException("Bad data format, unzip failed");
				}
				//System.out.println("Current index position = " + curPos + ", end position = " + endDocId);
				nextCounter++;
				return true;	
			}
			else
			{
                return false;
			}
        }
			
		private byte[] processCompressedDoc(byte[] compressed) throws java.util.zip.DataFormatException 
		{
			ByteArrayOutputStream unzipStream = new ByteArrayOutputStream();
			Inflater inflater = new Inflater();
			int nBytes;

			inflater.setInput(compressed, 0, compressed.length);
			while ((nBytes = inflater.inflate(compressionBuf)) > 0)
			{
				unzipStream.write(compressionBuf, 0, nBytes);
			}
			inflater.end();

			byte[] rawData = unzipStream.toByteArray();
			return rawData;
		}
		
        @Override
        public LongWritable getCurrentKey() throws IOException, InterruptedException
		{
            return key;
        }

        @Override
        public Text getCurrentValue() throws IOException, InterruptedException
		{
            return value;
        }

        @Override
        public float getProgress() throws IOException, InterruptedException
		{
            return (float)bytesRead / totalSize;
        }

        @Override
        public void close() throws IOException
		{
            IOUtils.closeStream(input);
			//System.out.println("Read documents from split: " + nextCounter);
        }
	}
	
	private List<Path> getDataFiles(JobContext context) throws IOException
	{
		List<Path> dataFileNames = new ArrayList<Path>();
		for (FileStatus status: listStatus(context))
		{
			Path dataPath = status.getPath();
			dataFileNames.add(dataPath);
		}
		return dataFileNames;
	}
	
	private ArrayList<Integer> readFileIndex(Path idxFilePath, Configuration conf) throws IOException
	{
		FileSystem fs = idxFilePath.getFileSystem(conf);
		ArrayList<Integer> idx = new ArrayList<Integer>();

        FSDataInputStream in = null;
        try
		{
            in = fs.open(idxFilePath);
			while (true)
			{
	            int docDataSize;
				try
				{
					docDataSize = Integer.reverseBytes(in.readInt());
					idx.add(docDataSize);
				}
				catch (EOFException e)
				{
					break;
				}
			}
        }
        finally
		{
            IOUtils.closeStream(in);
        }
		return idx;
	}
	
	@Override
    public List<InputSplit> getSplits(JobContext context) throws IOException {
        List<InputSplit> splits = new ArrayList<InputSplit>();
		long stdSplitSize = getStdSplitSize(context.getConfiguration());

		List<Path> dataFiles = getDataFiles(context);
		for (Path dataFilePath : dataFiles)
		{
			Path indexPath = dataFilePath.suffix(".idx");
			ArrayList<Integer> fileIndex = readFileIndex(indexPath, context.getConfiguration());			
			int posInIndex = 0;
			long offset = 0L;
			while (true)
			{
				long splitSize = 0L;
				int startPosInIndex = posInIndex;
				while (splitSize < stdSplitSize && posInIndex < fileIndex.size())
				{
					splitSize += fileIndex.get(posInIndex++);
				}
				int endPos = posInIndex - 1;
				IndexedSplit split = new IndexedSplit(dataFilePath, offset, splitSize, fileIndex,
					startPosInIndex, endPos);
				splits.add(split);
				//System.out.println("Added split with size " + splitSize + " bytes, startPos = " + split.getStartDocId() + ", endPos = " + split.getEndDocId());
				offset += splitSize;
				if (posInIndex >= fileIndex.size())
				{
					break;
				}
			}
		}

        return splits;
    }
	
	@Override
    public RecordReader<LongWritable, Text> createRecordReader(InputSplit split, TaskAttemptContext context)
            throws IOException, InterruptedException {
        GzipRecordReader reader = new GzipRecordReader();
        reader.initialize(split, context);
        return reader;
    }
	
	public static final String BYTES_PER_MAP = "mapreduce.input.indexedgz.bytespermap";
	
	public static long getStdSplitSize(Configuration conf)
	{
        return conf.getLong(BYTES_PER_MAP, 67108864);
    }
}