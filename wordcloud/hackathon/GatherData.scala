package com.identified.bvoneoffs.hackathon

import collection.mutable
import com.identified.bvoneoffs.common.ScalaDbUtils.TraversableResultSet
import com.identified.bvoneoffs.common.{FileUtils, UsefulDbCalls, DBConn}
import io.Source

object GatherData {
  val bartevId = 2803522
  val janId = 674095243
  val vladId = 646089725

  private[this] val logger = grizzled.slf4j.Logger[this.type]

  def getSqlGetJobHistory(userId: Long): String = {
    "select j.display_company_name, j.display_job_title_name, o.org_name, i.iic_name, j.description " +
      "   from user_jobs j " +
      "     left outer join orgs o on j.company_summary_id = o.id " +
      "     left outer join ic_iic_codes i on o.industry_id = i.iic " +
      "   where user_id = " + userId +
      "     order by start_date desc;"
  }
  def getSqlGetEducHistory(userId: Long): String = {
    "select e.display_university_name, e.display_degree_name, e.display_major_name, e.display_minor_name, o.org_name, m.major, mm.major as minor " +
      " 	from user_education e left outer join orgs o on e.university_id = o.id " +
      " 	 	left outer join cu_majors m on e.major_id  = m.id " +
      " 	 	left outer join cu_majors mm on e.minor_id = mm.id " +
      " 	where user_id = " + userId +
      " 	order by e.start_date desc;"
  }
  def getSqlFriendsId(userId: Long): String = "select friend_id from user_friends f where f.user_id = " + userId

  def getUserFriends(userId: Long): List[Long] = {
    val conn = DBConn("prd-db-slave").getConnection
    try {
      val rs = conn.prepareStatement(getSqlFriendsId(userId)).executeQuery()
      rs.map(r => r.getLong(1)).toList
    } catch {
      case e: Exception => logger.error("error getting user friends")
      List()
    }
  }

  def getUserText(userId: Long) = {
    val allIds = userId :: getUserFriends(userId)

    def loop(acc: List[String], userIds: List[Long]): List[String] = userIds match {
      case Nil => {
        //        logger.debug("in case nil")
        acc
      }
      case xs: List[Long] => {
        //        logger.debug("in case xs:List[Long]")
        val conn = DBConn("prd-db-slave").getConnection

        val jobsRs = conn.prepareCall(getSqlGetJobHistory(xs.head)).executeQuery
        conn.close()
        val jobs = jobsRs.map(r => UsefulDbCalls.convertResultSetRowToString(r)).toList
        //        logger.debug(xs.head + "\t" + jobs)
        loop(acc ::: jobs, xs.tail)
      }
      case _ => {
        logger.debug("In case _")
        acc
      }
    }

    loop(List(), allIds)
  }

  def getTextFromFile(fname: String): String = Source.fromFile(fname).mkString

  def countWords(text: String) = {
    val stopWords = ("a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at," +
      "be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from," +
      "get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,l" +
      "et,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other," +
      "our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there," +
      "these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom," +
      "why,will,with,would,yet,you,your").split(",").map(_.trim)

    val counts = mutable.Map.empty[String, Int]
    logger.info(text)
    val words = text.split("[ ,'\"!.(\\-)0-9/\t\n&%$#]+")
                .map(_.toLowerCase.trim)
                .filterNot(_ == "null")
                .filterNot(p => stopWords.contains(p))
    val cnts = words.groupBy(identity).mapValues(_.size)
    logger.info(words)
    logger.info(cnts)
    cnts
  }

  def formatCountForWordle(cnts: Map[String, Int]): String = {
    val cntsSorted = cnts.toList.sortBy(-_._2)
    cntsSorted.map(item => item._1 + ":" + item._2).mkString("\n")
  }

  def writeMapToJson(cnts: Map[String, Int], fname: String) {
    val m = cnts.toList.sortBy(-_._2)
    val m2 = m.map(item => "{\"text\":\"" + item._1 + "\",\"size\":" + item._2 + "}").mkString("[", ",", "]")
    FileUtils.writeStringToFile(m2, fname)
  }

  def main(args: Array[String]) {
    logger.debug("begin main")
    val bvFname = "/Users/bartev/PersonalDevelopment/sam_adams/hackathon/data/bartevsString.tsv"
    val bvWordleFname = "/Users/bartev/PersonalDevelopment/sam_adams/hackathon/data/bartevsStringForWordle.tsv"
    val bvFnameJson = "/Users/bartev/PersonalDevelopment/sam_adams/hackathon/data/bartevsStringJson.tsv"
    val bvFnameJsonWc = "/Users/bartev/Development/mine/bartev/testplay/wordcloud/public/data/bartevString.json"

    // TODO - uncomment these when want to get data from db again
    // Leave commented to just use saved data
//    val userJobEducString = getUserText(bartevId)
    val userJobEducString = getUserText(vladId)
    FileUtils.writeListToTSV(userJobEducString, bvFname, addTimeStamp = false, append = false, header = "", suffix = "")
    val s = getTextFromFile(bvFname)

    val testText = "Hello, hello, my name is 999 Bartev!"
    val m1 = countWords(userJobEducString.mkString(" ")).take(150)
    val m = countWords(s).take(200)
    println(m.take(10))
    userJobEducString.take(10).foreach(println(_))

    //    val m = countWords(testText).take(50)

    writeMapToJson(m, bvFnameJsonWc)

    //    val formattedText = formatCountForWordle(countWords(s))
    //    FileUtils.writeStringToFile(formattedText, bvWordleFname)
  }
}
