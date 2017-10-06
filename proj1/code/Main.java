import java.io.IOException;
import java.util.ArrayList;

public class Main {
	
	private static String inputFile = "input.data";
	private static String outputFile = "ourData.data";

	public static void main(String[] args) {
		
		if (args.length == 2) {
			inputFile = args[0];
			outputFile = args[1];
		}
		
		InputParser input = new InputParser(inputFile);
		input.readInputFromFile();
		
		ArrayList<Classification> classificationList = input.getClassificationList();
		DataCreator dataCreator = new DataCreator();
		
		ArrayList<DataObject> dataList = dataCreator.createData(classificationList);
		
		OutputWriter output = new OutputWriter();
		
		try {
			output.writeDataToFile(outputFile, dataList);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
