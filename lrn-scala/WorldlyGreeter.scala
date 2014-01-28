//
//	WorldlyGreeter.scala
//	/Users/bartev/Development/san-tan/lrn-scala/WorldlyGreeter.scala
//

// The WorldlyGreeter class
class WorldlyGreeter(greeting: String) {
	def greet() = {
		val worldlyGreeting = WorldlyGreeter.worldify(greeting)	
		println(worldlyGreeting)
	}
}

// the WorldlyGreeter companion object
object WorldlyGreeter {
	def worldify(s: String) = {
		s + ", world!"
	}
}
