package com.identified.bvoneoffs.hackathon.examples

import swing._
import swing.event._

object TempConverter extends SimpleSwingApplication {
  def trunc(x:Double, n:Int) = math.round(x * math.pow(10.0, n))/math.pow(10.0, n)
  def top = new MainFrame() {
    title = "Celsius/Farenheit Converter"

    object celsius extends TextField(columns = 5)

    object farenheit extends TextField(columns = 5)

    contents = new FlowPanel {
      contents += celsius
      contents += new Label(" Celsius = ")
      contents += farenheit
      contents += new Label(" Farenheit")
      border = Swing.EmptyBorder(15, 10, 10, 10)
    }
    listenTo(celsius, farenheit)
    reactions += {
      case EditDone(`farenheit`) =>
        val f = farenheit.text.toDouble
        val c = (f - 32.0) * 5 / 9
        celsius.text = trunc(c, 2).toString
      case EditDone(`celsius`) =>
        val c = celsius.text.toDouble
        val f = c * 9 / 5 + 32.0
        farenheit.text = trunc(f, 2).toString
    }
  }
}
