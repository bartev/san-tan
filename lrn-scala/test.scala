// Comparte function and method

// http://jim-mcbeath.blogspot.com/2009/05/scala-functions-vs-methods.html

class test() {
	def m1(x:Int) = x + 3
	val f1 = (x:Int) => x + 3
	val f2 = m1 _
}