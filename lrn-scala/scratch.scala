val list = List("hello", "world")

list match {
	case Nil => "was an empty list"
	case x::xs => "head was " + x + ", tail was " + xs
}


object match {
	case Address(Name(first, last), street, city, state, zip) =>
		println(last + ", " + zip)
	case _ => println("not an address") // the default case
}

case class Fruit(name: String, color: String, citrus: Boolean) 
val apple = Fruit("Apple", "green", false)
val pear = apple.copy(name = "Pear")
val lime = apple.copy(name = "Lime", citrus = true)


class C {
	var acc = 0
	def minc = {acc += 1}
	val finc = { () => acc += 1}
}

val c = new C
c.minc