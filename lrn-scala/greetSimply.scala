//
//	greetSimply.scala
//	/Users/bartev/Development/san-tan/lrn-scala/greetSimply.scala
//

class SimpleGreeter {
	// Field
	val greeting = "Hello, world!"
	// Method
	def greet() = println(greeting)
}

// initialize a val with a new instance of SimpleGreeter
val g = new SimpleGreeter
g.greet()