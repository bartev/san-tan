//
//	greegreetFancily.scala
//	/Users/bartev/Development/san-tan/lrn-scala/greegreetFancily.scala
//

class FancyGreeter(greeting: String) {
	def greet() = println(greeting)
}

val g = new FancyGreeter("Salutations, world")
g.greet