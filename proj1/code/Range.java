public class Range {
	
	private double lowerBound;
	private double upperBound;
	
	public Range(double lower, double upper) {
		lowerBound = lower;
		upperBound = upper;
	}
	
	public Range() {}
	
	public boolean inRange(double x) {
		
		if (x >= lowerBound && x < upperBound) {
			return true;
		}
		
		return false;
	}

	public double getLowerBound() {
		return lowerBound;
	}

	public void setLowerBound(double lowerBound) {
		this.lowerBound = lowerBound;
	}

	public double getUpperBound() {
		return upperBound;
	}

	public void setUpperBound(double upperBound) {
		this.upperBound = upperBound;
	}
}
