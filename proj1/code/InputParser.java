import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class InputParser {
	
	private ArrayList<Classification> classificationList;
	private String fileName;
	
	public InputParser(String fileName) {
		this.fileName = fileName;
		classificationList = new ArrayList<>();
	}
	
	/*
	 * Let the input file be in the form:
	 * Class 1
	 * P,Q [where P is the lower bound of attribute 1 and Q is the upper bound of attribute 1]
	 * P,Q
	 * ... [for all M attributes]
	 * P,Q
	 * Class 2
	 * ...
	 * Class N
	 */
	public void readInputFromFile() {

		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
			
			Classification classification = new Classification();
			
			String line = br.readLine();
			
			while (line != null) {
				String[] delimitedLine = line.split(",");

				// New Classification
				if (delimitedLine.length == 1) {
					
					if (!classification.getClassification().equals("")) {
						classificationList.add(classification);
					}
					
					classification = new Classification();
					classification.setClassification(line);
				} else if (delimitedLine.length == 2) {
					Range range = new Range(Double.parseDouble(delimitedLine[0]), Double.parseDouble(delimitedLine[1]));
					
					classification.addAttributeRange(range);
				} else {
					throw new IllegalArgumentException("Incorrectly formatted input file.");
				}
			
				line = br.readLine();
			}
			
			if (!classification.getClassification().equals("")) {
				classificationList.add(classification);
			}
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public ArrayList<Classification> getClassificationList() {
		return classificationList;
	}

	public void setClassificationList(ArrayList<Classification> classificationList) {
		this.classificationList = classificationList;
	}
}
