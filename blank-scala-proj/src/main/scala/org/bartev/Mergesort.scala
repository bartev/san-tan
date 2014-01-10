package org.bartev

/**
 * Created with IntelliJ IDEA.
 * User: bartev
 * Date: 5/10/13
 * Time: 2:27 PM
 * To change this template use File | Settings | File Templates.
 */
object Mergesort {
  // Copied from ?
  // This version allows you to define a function for 'less'
  def msort[T](less: (T, T) => Boolean)(xs: List[T]): List[T] = {
    // combine step
    // Theta(n) time
    def merge(xs: List[T], ys: List[T]): List[T] = (xs, ys) match {
      case (Nil, _) => ys
      case (_, Nil) => xs
      case (x :: xs1, y :: ys1) =>
        if (less(x, y)) x :: merge(xs1, ys)
        else y :: merge(xs, ys1)
    }

    val n = xs.length / 2
    if (n == 0) xs
    else {
      val (ys, zs) = xs splitAt (n) // Split the list in half
      merge(msort(less)(ys), msort(less)(zs))
    }
  }
}
