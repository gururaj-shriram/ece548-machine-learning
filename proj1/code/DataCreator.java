import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class DataCreator {
	
	private static final int NUM_DATA_ELEMENTS = 100;
	
	/**
	 * @param classificationList
	 * @return
	 */
	public ArrayList<DataObject> createData(ArrayList<Classification> classificationList) {
		
		ArrayList<DataObject> dataList = new ArrayList<>();
		int numElementsPerClassification = NUM_DATA_ELEMENTS / classificationList.size();
		int numAttributes = classificationList.get(0).getAttributeRanges().size();
		
		// For each class
		for (int i = 0; i < classificationList.size(); i++) {
			Classification classification = classificationList.get(i);
			
			// Create N / #_of_classes data elements
			for (int j = 0; j < numElementsPerClassification; j++) {
				DataObject data = new DataObject();
				data.setClassification(classification.getClassification());
				
				// Generate a value per attribute
				for (int k = 0; k < numAttributes; k++) {
					Range range = classification.getAttributeRanges().get(k);
					double attributeValue = ThreadLocalRandom.current().nextDouble(range.getLowerBound(), range.getUpperBound());
					
					attributeValue = (double) Math.round(attributeValue * 100) / 100;
					data.addAttribute(attributeValue);
				}
				
				dataList.add(data);
			}
		}
		
		return dataList;
	}
}
