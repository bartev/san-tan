package org.bartev

/**
 * Created with IntelliJ IDEA.
 * User: bartev
 * Date: 5/10/13
 * Time: 2:02 PM
 * To change this template use File | Settings | File Templates.
 */
object Quicksort {
  def qsort[T <% Ordered[T]](xs: List[T]): List[T] = {
    xs match {
      case Nil => Nil
      case x :: xs1 =>
        // Here the pivot is the first element of the list
        val (before, after) = xs1.partition(_ < x)
        qsort(before) ++ (x :: qsort(after))
    }
  }

  def sort[T <% Ordered[T]](xs: List[T]): List[T] = {
    if (xs.isEmpty || xs.tail.isEmpty) xs
    else {
      // Here the pivot is taken as the midpoint of the list
      val pivot = xs(xs.length / 2)
      var lows: List[T] = Nil
      var mids: List[T] = Nil
      var highs: List[T] = Nil

      // divide
      for (item <- xs) {
        if (item == pivot) mids = item :: mids
        else if (item < pivot) lows = item :: lows
        else highs = item :: highs
      }
      // conquer & combine
      sort(lows) ::: mids ::: sort(highs)
    }
  }

  def sort2(xs: Array[Int]): Array[Int] = {
    if (xs.length < 2) xs
    else {
      val pivot = xs(xs.length / 2)
      val (lows, midHighs) = xs.partition(_ < pivot)
      val (mids, highs) = midHighs.partition(_ == pivot)
      Array.concat(sort2(lows), mids, sort2(highs))
    }
  }

  def sort3(xs: Array[Int]) = {
    def swap(i: Int, j: Int) {
      val t = xs(i)
      xs(i) = xs(j)
      xs(j) = t
    }
    def sortx(l: Int, r: Int) {
      val pivot = xs((l + r) / 2)
      var i = l
      var j = r
      while (i <= j) {
        while (xs(i) < pivot) i += 1
        while (xs(j) > pivot) j -= 1

        if (i <= j) {
          swap(i, j)
          i += 1
          j -= 1
        }
      }
      if (l < j) sortx(l, j)
      if (j < r) sortx(i, r)
    }
    sortx(0, xs.length - 1)
  }

  def sort4(xs: Array[Int]): Array[Int] = {
    if (xs.length <= 1) xs
    else {
      val pivot = xs(xs.length / 2)
      Array.concat(
        sort4(xs.filter(_ < pivot)),
        xs.filter(_ == pivot),
        sort4(xs.filter(_ > pivot))
      )
    }
  }
}