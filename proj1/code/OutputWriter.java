import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

public class OutputWriter {
	
	public void writeDataToFile(String fileName, ArrayList<DataObject> dataList) throws UnsupportedEncodingException, FileNotFoundException, IOException {
			
			try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(
		              new FileOutputStream(fileName), "utf-8"))) {
				
				for (DataObject data : dataList) {
					
					for (double attributeValue : data.getAttributes()) {
						writer.write(attributeValue + ",");
					}
					
					writer.write(data.getClassification());
					writer.newLine();
				}
			}
		}
}
