// In greetCarefully

class CarefulGreeter(greeting: String) {
	if (greeting == null) {
		throw new NullPointerException("greeting was null")
	}
	
	def greet() = println(greeting)	
}

new CarefulGreeter(null)

