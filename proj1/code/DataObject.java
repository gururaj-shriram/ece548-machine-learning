import java.util.ArrayList;

/**
 * Encapsulates data with class name and attribute values
 *
 */
public class DataObject {
	
	private ArrayList<Double> attributes;
	private String classification;
	
	public DataObject() {
		attributes = new ArrayList<Double>();
		classification = "";
	}
	
	public void addAttribute(double attribute) {
		attributes.add(attribute);
	}
	
	public void removeAttribute(double attribute) {
		attributes.remove(attribute);
	}

	public ArrayList<Double> getAttributes() {
		return attributes;
	}

	public void setAttributes(ArrayList<Double> attributes) {
		this.attributes = attributes;
	}

	public String getClassification() {
		return classification;
	}

	public void setClassification(String classification) {
		this.classification = classification;
	}

}
