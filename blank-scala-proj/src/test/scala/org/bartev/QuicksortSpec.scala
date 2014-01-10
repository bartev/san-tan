package org.bartev

import org.scalatest.FlatSpec
import org.scalatest.matchers.ShouldMatchers

/**
 * Created with IntelliJ IDEA.
 * User: bartev
 * Date: 5/10/13
 * Time: 2:04 PM
 * To change this template use File | Settings | File Templates.
 */
class QuicksortSpec extends FlatSpec with ShouldMatchers {
  def initAll = new {
    val intList = List(9, 4, 5, 1, 2, 0, 3, 8, 7, 6)
    val stringList = List("Berlin", "Paris", "Barcelona", "Amsterdam")
    val intArray = Array(9, 4, 5, 1, 2, 0, 3, 8, 7, 6)
    val repIntList = List(9, 0, 5, 5, 4)
  }
  "repIntList" should "be sorted" in {
    Quicksort.qsort(initAll.repIntList) should equal (List(0, 4, 5, 5, 9))
  }
  "intList" should "be sorted" in {
    Quicksort.qsort(initAll.intList) should equal ((0 to 9).toList)
  }

  it should "be sorted2" in {
    Quicksort.sort(initAll.intList) should equal ((0 to 9).toList)
  }
  it should "be sorted3" in {
    Quicksort.sort2(initAll.intArray) should equal ((0 to 9).toArray)
  }

}
