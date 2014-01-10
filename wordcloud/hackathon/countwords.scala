// Downloaded from
// http://mkaz.com/dataviz/obama-2012-budget-in-a-word-cloud#respond
// On 2013-03-13

import scala.collection.mutable.HashMap

val file = "budget.txt"

// download from InfoChimps
// http://infochimps.com/datasets/list-of-english-stopwords
val stopwords = "stopwords.txt" 

// slurp text file in
var text = io.Source.fromFile(file).mkString
 
// remove punctation
text = text.replace(".", "")
text = text.replace(",", "")
text = text.replace(";", "")
text = text.replace("-", "")
text = text.replace("?", "")
text = text.replace("\"", "")
text = text.replace("\n", "")

// convert to lowercase
text = text.toLowerCase

// chunk text to words
val allwords = text.split(' ').toList

// load up stop words
val stop = io.Source.fromFile(stopwords).getLines().toList
val words = allwords filter { w => !stop.contains(w) }

val wmap = new HashMap[String, Int]()

// group words together
words foreach { word =>
  val c = wmap.getOrElse(word, 0)
  wmap.update(word, c+1)
}

// print out sorted list
wmap.toList sortBy {_._2} foreach {
  case (key, value) =>
    println(key + ": " + value)
}

// feed word list into www.wordle.net