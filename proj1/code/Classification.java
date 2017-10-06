import java.util.ArrayList;

/**
 * Stores a class along with ranges of attributes
 */
public class Classification {
	
	private String classification;
	private ArrayList<Range> attributeRanges;
	
	public Classification() {
		attributeRanges = new ArrayList<>();
		classification = "";
	}
	
	public String getClassification() {
		return classification;
	}
	
	public void addAttributeRange(Range range) {
		attributeRanges.add(range);
	}
	
	public void setClassification(String classification) {
		this.classification = classification;
	}
	
	public ArrayList<Range> getAttributeRanges() {
		return attributeRanges;
	}
	
	public void setAttributeRanges(ArrayList<Range> attributeRanges) {
		this.attributeRanges = attributeRanges;
	}

}
